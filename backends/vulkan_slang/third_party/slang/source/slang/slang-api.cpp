// slang-api.cpp

#include "../core/slang-performance-profiler.h"
#include "../core/slang-platform.h"
#include "../core/slang-rtti-info.h"
#include "../core/slang-shared-library.h"
#include "../core/slang-signal.h"
#include "../slang-record-replay/record/slang-global-session.h"
#include "../slang-record-replay/util/record-utility.h"
#include "slang-capability.h"
#include "slang-compiler.h"
#include "slang-internal.h"
#include "slang-repro.h"

// implementation of C interface

SLANG_API SlangSession* spCreateSession(const char*)
{
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    if (SLANG_FAILED(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef())))
    {
        return nullptr;
    }
    // Will be returned with a refcount of 1
    return globalSession.detach();
}

// Attempt to load a previously compiled builtin module from the same file system location as the
// slang dll. Returns SLANG_OK when the cache is sucessfully loaded. Also returns the filename to
// the builtin module cache and the timestamp of current slang dll.
SlangResult tryLoadBuiltinModuleFromCache(
    slang::IGlobalSession* globalSession,
    slang::BuiltinModuleName builtinModuleName,
    Slang::String& outCachePath,
    uint64_t& outTimestamp)
{
    auto fileName =
        Slang::SharedLibraryUtils::getSharedLibraryFileName((void*)slang_createGlobalSession);
    uint64_t currentLibTimestamp =
        Slang::SharedLibraryUtils::getSharedLibraryTimestamp((void*)slang_createGlobalSession);
    auto dirName = Slang::Path::getParentDirectory(fileName);
    auto cacheFileName = Slang::Path::combine(
        dirName,
        Slang::String("slang-") + Slang::getBuiltinModuleNameStr(builtinModuleName) +
            "-module.bin");
    outTimestamp = currentLibTimestamp;
    outCachePath = cacheFileName;
    if (currentLibTimestamp == 0)
    {
        return SLANG_FAIL;
    }
    Slang::ScopedAllocation cacheData;
    SLANG_RETURN_ON_FAIL(Slang::File::readAllBytes(cacheFileName, cacheData));

    // The first 8 bytes stores the timestamp of the slang dll that created this core module cache.
    if (cacheData.getSizeInBytes() < sizeof(uint64_t))
        return SLANG_FAIL;
    auto cacheTimestamp = *(uint64_t*)(cacheData.getData());
    if (cacheTimestamp != currentLibTimestamp)
        return SLANG_FAIL;
    SLANG_RETURN_ON_FAIL(globalSession->loadBuiltinModule(
        builtinModuleName,
        (uint8_t*)cacheData.getData() + sizeof(uint64_t),
        cacheData.getSizeInBytes() - sizeof(uint64_t)));
    return SLANG_OK;
}

// Attempt to load a precompiled builtin module from slang-xxx-module.dll.
SlangResult tryLoadBuiltinModuleFromDLL(
    slang::IGlobalSession* globalSession,
    slang::BuiltinModuleName builtinModuleName)
{
    Slang::String moduleFileName =
        Slang::String("slang-") + Slang::getBuiltinModuleNameStr(builtinModuleName) + "-module";

    Slang::SharedLibrary::Handle libHandle = nullptr;

    SLANG_RETURN_ON_FAIL(Slang::SharedLibrary::load(moduleFileName.getBuffer(), libHandle));
    if (!libHandle)
        return SLANG_FAIL;
    void* ptr = Slang::SharedLibrary::findSymbolAddressByName(libHandle, "slang_getEmbeddedModule");
    if (!ptr)
        return SLANG_FAIL;
    typedef ISlangBlob*(GetEmbeddedModuleFunc)();
    auto getEmbeddedModule = (GetEmbeddedModuleFunc*)ptr;
    auto blob = getEmbeddedModule();
    SLANG_RETURN_ON_FAIL(globalSession->loadBuiltinModule(
        builtinModuleName,
        (uint8_t*)blob->getBufferPointer(),
        blob->getBufferSize()));
    return SLANG_OK;
}

SlangResult trySaveBuiltinModuleToCache(
    slang::IGlobalSession* globalSession,
    slang::BuiltinModuleName builtinModuleName,
    const Slang::String& cacheFilename,
    uint64_t dllTimestamp)
{
    if (dllTimestamp != 0 && cacheFilename.getLength() != 0)
    {
        Slang::ComPtr<ISlangBlob> coreModuleBlobPtr;
        SLANG_RETURN_ON_FAIL(globalSession->saveBuiltinModule(
            builtinModuleName,
            SLANG_ARCHIVE_TYPE_RIFF_LZ4,
            coreModuleBlobPtr.writeRef()));

        Slang::FileStream fileStream;
        SLANG_RETURN_ON_FAIL(fileStream.init(cacheFilename, Slang::FileMode::Create));

        SLANG_RETURN_ON_FAIL(fileStream.write(&dllTimestamp, sizeof(dllTimestamp)));
        SLANG_RETURN_ON_FAIL(fileStream.write(
            coreModuleBlobPtr->getBufferPointer(),
            coreModuleBlobPtr->getBufferSize()))
    }

    return SLANG_OK;
}

SLANG_API SlangResult
slang_createGlobalSession(SlangInt apiVersion, slang::IGlobalSession** outGlobalSession)
{
    SlangGlobalSessionDesc desc = {};
    desc.apiVersion = (uint32_t)apiVersion;
    return slang_createGlobalSession2(&desc, outGlobalSession);
}

SLANG_API SlangResult slang_createGlobalSessionImpl(
    const SlangGlobalSessionDesc* desc,
    const Slang::GlobalSessionInternalDesc* internalDesc,
    slang::IGlobalSession** outGlobalSession)
{
    Slang::ComPtr<slang::IGlobalSession> globalSession;

#ifdef SLANG_ENABLE_IR_BREAK_ALLOC
    // Set inst debug alloc counter to 0 so IRInsts for core module always starts from a large
    // value.
    Slang::_debugGetIRAllocCounter() = 0x80000000;
#endif

    SLANG_RETURN_ON_FAIL(
        slang_createGlobalSessionWithoutCoreModule(desc->apiVersion, globalSession.writeRef()));

    // If we have the embedded core module, load from that, else compile it
    ISlangBlob* coreModuleBlob = slang_getEmbeddedCoreModule();
    if (coreModuleBlob)
    {
        SLANG_RETURN_ON_FAIL(globalSession->loadCoreModule(
            coreModuleBlob->getBufferPointer(),
            coreModuleBlob->getBufferSize()));
    }
    else
    {
        Slang::String cacheFilename;
        uint64_t dllTimestamp = 0;
        SlangResult loadFromCacheResult = SLANG_FAIL;
        if (!internalDesc->isBootstrap)
        {
            loadFromCacheResult = tryLoadBuiltinModuleFromCache(
                globalSession,
                slang::BuiltinModuleName::Core,
                cacheFilename,
                dllTimestamp);
        }
        if (loadFromCacheResult != SLANG_OK)
        {
            // Compile std lib from embeded source.
            SLANG_RETURN_ON_FAIL(
                globalSession->compileBuiltinModule(slang::BuiltinModuleName::Core, 0));
            // Store the compiled core module to cache file.
            trySaveBuiltinModuleToCache(
                globalSession,
                slang::BuiltinModuleName::Core,
                cacheFilename,
                dllTimestamp);
        }
    }

    if (desc->enableGLSL)
    {
        Slang::String cacheFilename;
        uint64_t dllTimestamp = 0;
        SlangResult loadFromCacheResult = SLANG_FAIL;
        if (!internalDesc->isBootstrap)
        {
            loadFromCacheResult =
                tryLoadBuiltinModuleFromDLL(globalSession, slang::BuiltinModuleName::GLSL);
            if (SLANG_FAILED(loadFromCacheResult))
            {
                loadFromCacheResult = tryLoadBuiltinModuleFromCache(
                    globalSession,
                    slang::BuiltinModuleName::GLSL,
                    cacheFilename,
                    dllTimestamp);
            }
        }
        if (SLANG_FAILED(loadFromCacheResult))
        {
            SLANG_RETURN_ON_FAIL(
                globalSession->compileBuiltinModule(slang::BuiltinModuleName::GLSL, 0));

            // Store the compiled core module to cache file.
            trySaveBuiltinModuleToCache(
                globalSession,
                slang::BuiltinModuleName::GLSL,
                cacheFilename,
                dllTimestamp);
        }
    }

    // Check if the SLANG_CAPTURE_ENABLE_ENV is enabled
    if (SlangRecord::isRecordLayerEnabled())
    {
        SlangRecord::GlobalSessionRecorder* globalSessionRecorder =
            new SlangRecord::GlobalSessionRecorder(desc, globalSession.detach());
        Slang::ComPtr<SlangRecord::GlobalSessionRecorder> result(globalSessionRecorder);
        *outGlobalSession = result.detach();
    }
    else
    {
        *outGlobalSession = globalSession.detach();
    }

#ifdef SLANG_ENABLE_IR_BREAK_ALLOC
    // Reset inst debug alloc counter to 0 so IRInsts for user code always starts from 0.
    Slang::_debugGetIRAllocCounter() = 0;
#endif

    return SLANG_OK;
}

SLANG_API SlangResult slang_createGlobalSession2(
    const SlangGlobalSessionDesc* desc,
    slang::IGlobalSession** outGlobalSession)
{
    Slang::GlobalSessionInternalDesc internalDesc = {};
    return slang_createGlobalSessionImpl(desc, &internalDesc, outGlobalSession);
}

SLANG_API void slang_shutdown()
{
    Slang::PerformanceProfiler::getProfiler()->dispose();
    Slang::SPIRVCoreGrammarInfo::freeEmbeddedGrammerInfo();
    Slang::RttiInfo::deallocateAll();
    Slang::freeCapabilityDefs();
}

SLANG_API SlangResult slang_createGlobalSessionWithoutCoreModule(
    SlangInt apiVersion,
    slang::IGlobalSession** outGlobalSession)
{
    if (apiVersion != 0)
        return SLANG_E_NOT_IMPLEMENTED;

    // Create the session
    Slang::Session* globalSession = new Slang::Session();
    // Put an interface ref on it
    Slang::ComPtr<slang::IGlobalSession> result(globalSession);

    // Initialize it
    globalSession->init();

    *outGlobalSession = result.detach();
    return SLANG_OK;
}

SLANG_API const char* slang_getLastInternalErrorMessage()
{
    return Slang::getLastSignalMessage();
}

SLANG_API void spDestroySession(SlangSession* inSession)
{
    if (!inSession)
        return;

    Slang::Session* session = Slang::asInternal(inSession);
    // It is assumed there is only a single reference on the session (the one placed
    // with spCreateSession) if this function is called
    SLANG_ASSERT(session->debugGetReferenceCount() == 1);
    // Release
    session->release();
}

SLANG_API const char* spGetBuildTagString()
{
    return Slang::getBuildTagString();
}

SLANG_API void spAddBuiltins(
    SlangSession* session,
    char const* sourcePath,
    char const* sourceString)
{
    session->addBuiltins(sourcePath, sourceString);
}

SLANG_API void spSessionSetSharedLibraryLoader(
    SlangSession* session,
    ISlangSharedLibraryLoader* loader)
{
    session->setSharedLibraryLoader(loader);
}

SLANG_API ISlangSharedLibraryLoader* spSessionGetSharedLibraryLoader(SlangSession* session)
{
    return session->getSharedLibraryLoader();
}

SLANG_API SlangResult
spSessionCheckCompileTargetSupport(SlangSession* session, SlangCompileTarget target)
{
    return session->checkCompileTargetSupport(target);
}

SLANG_API SlangResult
spSessionCheckPassThroughSupport(SlangSession* session, SlangPassThrough passThrough)
{
    return session->checkPassThroughSupport(passThrough);
}

SLANG_API SlangCompileRequest* spCreateCompileRequest(SlangSession* session)
{
    slang::ICompileRequest* request = nullptr;
    // Will return with suitable ref count
    SLANG_ALLOW_DEPRECATED_BEGIN
    session->createCompileRequest(&request);
    SLANG_ALLOW_DEPRECATED_END
    return request;
}

SLANG_API SlangProfileID spFindProfile(SlangSession* session, char const* name)
{
    return session->findProfile(name);
}

SLANG_API SlangCapabilityID spFindCapability(SlangSession* session, char const* name)
{
    return session->findCapability(name);
}

/* !!!!!!!!!!!!!!!!!!SlangCompileRequest API!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/*!
@brief Destroy a compile request.
*/
SLANG_API void spDestroyCompileRequest(slang::ICompileRequest* request)
{
    if (request)
    {
        request->release();
    }
}

/* All other functions just call into the ICompileResult interface. */

SLANG_API void spSetFileSystem(slang::ICompileRequest* request, ISlangFileSystem* fileSystem)
{
    SLANG_ASSERT(request);
    request->setFileSystem(fileSystem);
}

SLANG_API void spSetCompileFlags(slang::ICompileRequest* request, SlangCompileFlags flags)
{
    SLANG_ASSERT(request);
    request->setCompileFlags(flags);
}

SLANG_API SlangCompileFlags spGetCompileFlags(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    return request->getCompileFlags();
}

SLANG_API void spSetDumpIntermediates(slang::ICompileRequest* request, int enable)
{
    SLANG_ASSERT(request);
    request->setDumpIntermediates(enable);
}

SLANG_API void spSetDumpIntermediatePrefix(slang::ICompileRequest* request, const char* prefix)
{
    SLANG_ASSERT(request);
    request->setDumpIntermediatePrefix(prefix);
}

SLANG_API void spSetLineDirectiveMode(slang::ICompileRequest* request, SlangLineDirectiveMode mode)
{
    SLANG_ASSERT(request);
    request->setLineDirectiveMode(mode);
}

SLANG_API void spSetTargetForceGLSLScalarBufferLayout(
    slang::ICompileRequest* request,
    int targetIndex,
    bool forceScalarLayout)
{
    SLANG_ASSERT(request);
    request->setTargetForceGLSLScalarBufferLayout(targetIndex, forceScalarLayout);
}

SLANG_API void spSetTargetUseMinimumSlangOptimization(
    slang::ICompileRequest* request,
    int targetIndex,
    bool val)
{
    SLANG_ASSERT(request);
    request->setTargetUseMinimumSlangOptimization(targetIndex, val);
}

SLANG_API void spSetIgnoreCapabilityCheck(slang::ICompileRequest* request, bool ignore)
{
    SLANG_ASSERT(request);
    request->setIgnoreCapabilityCheck(ignore);
}

SLANG_API void spSetTargetLineDirectiveMode(
    slang::ICompileRequest* request,
    int targetIndex,
    SlangLineDirectiveMode mode)
{
    SLANG_ASSERT(request);
    request->setTargetLineDirectiveMode(targetIndex, mode);
}

SLANG_API void spSetCommandLineCompilerMode(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    request->setCommandLineCompilerMode();
}

SLANG_API void spSetCodeGenTarget(slang::ICompileRequest* request, SlangCompileTarget target)
{
    SLANG_ASSERT(request);
    request->setCodeGenTarget(target);
}

SLANG_API int spAddCodeGenTarget(slang::ICompileRequest* request, SlangCompileTarget target)
{
    SLANG_ASSERT(request);
    return request->addCodeGenTarget(target);
}

SLANG_API void spSetTargetProfile(
    slang::ICompileRequest* request,
    int targetIndex,
    SlangProfileID profile)
{
    SLANG_ASSERT(request);
    request->setTargetProfile(targetIndex, profile);
}

SLANG_API void spSetTargetFlags(
    slang::ICompileRequest* request,
    int targetIndex,
    SlangTargetFlags flags)
{
    SLANG_ASSERT(request);
    request->setTargetFlags(targetIndex, flags);
}

SLANG_API void spSetTargetFloatingPointMode(
    slang::ICompileRequest* request,
    int targetIndex,
    SlangFloatingPointMode mode)
{
    SLANG_ASSERT(request);
    request->setTargetFloatingPointMode(targetIndex, mode);
}

SLANG_API void spAddTargetCapability(
    slang::ICompileRequest* request,
    int targetIndex,
    SlangCapabilityID capability)
{
    SLANG_ASSERT(request);
    request->addTargetCapability(targetIndex, capability);
}

SLANG_API void spSetMatrixLayoutMode(slang::ICompileRequest* request, SlangMatrixLayoutMode mode)
{
    SLANG_ASSERT(request);
    request->setMatrixLayoutMode(mode);
}

SLANG_API void spSetTargetMatrixLayoutMode(
    slang::ICompileRequest* request,
    int targetIndex,
    SlangMatrixLayoutMode mode)
{
    SLANG_ASSERT(request);
    request->setTargetMatrixLayoutMode(targetIndex, mode);
}

SLANG_API void spSetDebugInfoLevel(slang::ICompileRequest* request, SlangDebugInfoLevel level)
{
    SLANG_ASSERT(request);
    request->setDebugInfoLevel(level);
}

SLANG_API void spSetDebugInfoFormat(slang::ICompileRequest* request, SlangDebugInfoFormat format)
{
    SLANG_ASSERT(request);
    request->setDebugInfoFormat(format);
}

SLANG_API void spSetOptimizationLevel(slang::ICompileRequest* request, SlangOptimizationLevel level)
{
    SLANG_ASSERT(request);
    request->setOptimizationLevel(level);
}

SLANG_API void spSetOutputContainerFormat(
    slang::ICompileRequest* request,
    SlangContainerFormat format)
{
    SLANG_ASSERT(request);
    request->setOutputContainerFormat(format);
}

SLANG_API void spSetPassThrough(slang::ICompileRequest* request, SlangPassThrough passThrough)
{
    SLANG_ASSERT(request);
    request->setPassThrough(passThrough);
}

SLANG_API void spSetDiagnosticCallback(
    slang::ICompileRequest* request,
    SlangDiagnosticCallback callback,
    void const* userData)
{
    SLANG_ASSERT(request);
    request->setDiagnosticCallback(callback, userData);
}

SLANG_API void spSetWriter(
    slang::ICompileRequest* request,
    SlangWriterChannel chan,
    ISlangWriter* writer)
{
    SLANG_ASSERT(request);
    request->setWriter(chan, writer);
}

SLANG_API ISlangWriter* spGetWriter(slang::ICompileRequest* request, SlangWriterChannel chan)
{
    SLANG_ASSERT(request);
    return request->getWriter(chan);
}

SLANG_API void spAddSearchPath(slang::ICompileRequest* request, const char* path)
{
    SLANG_ASSERT(request);
    request->addSearchPath(path);
}

SLANG_API void spAddPreprocessorDefine(
    slang::ICompileRequest* request,
    const char* key,
    const char* value)
{
    SLANG_ASSERT(request);
    request->addPreprocessorDefine(key, value);
}

SLANG_API char const* spGetDiagnosticOutput(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    return request->getDiagnosticOutput();
}

SLANG_API SlangResult
spGetDiagnosticOutputBlob(slang::ICompileRequest* request, ISlangBlob** outBlob)
{
    SLANG_ASSERT(request);
    return request->getDiagnosticOutputBlob(outBlob);
}

// New-fangled compilation API

SLANG_API int spAddTranslationUnit(
    slang::ICompileRequest* request,
    SlangSourceLanguage language,
    char const* inName)
{
    SLANG_ASSERT(request);
    return request->addTranslationUnit(language, inName);
}

SLANG_API void spSetDefaultModuleName(
    slang::ICompileRequest* request,
    const char* defaultModuleName)
{
    SLANG_ASSERT(request);
    request->setDefaultModuleName(defaultModuleName);
}

SLANG_API SlangResult spAddLibraryReference(
    slang::ICompileRequest* request,
    const char* basePath,
    const void* libData,
    size_t libDataSize)
{
    SLANG_ASSERT(request);
    return request->addLibraryReference(basePath, libData, libDataSize);
}

SLANG_API void spTranslationUnit_addPreprocessorDefine(
    slang::ICompileRequest* request,
    int translationUnitIndex,
    const char* key,
    const char* value)
{
    SLANG_ASSERT(request);
    request->addTranslationUnitPreprocessorDefine(translationUnitIndex, key, value);
}

SLANG_API void spAddTranslationUnitSourceFile(
    slang::ICompileRequest* request,
    int translationUnitIndex,
    char const* path)
{
    SLANG_ASSERT(request);
    request->addTranslationUnitSourceFile(translationUnitIndex, path);
}

SLANG_API void spAddTranslationUnitSourceString(
    slang::ICompileRequest* request,
    int translationUnitIndex,
    char const* path,
    char const* source)
{
    SLANG_ASSERT(request);
    request->addTranslationUnitSourceString(translationUnitIndex, path, source);
}

SLANG_API void spAddTranslationUnitSourceStringSpan(
    slang::ICompileRequest* request,
    int translationUnitIndex,
    char const* path,
    char const* sourceBegin,
    char const* sourceEnd)
{
    SLANG_ASSERT(request);
    request->addTranslationUnitSourceStringSpan(translationUnitIndex, path, sourceBegin, sourceEnd);
}

SLANG_API void spAddTranslationUnitSourceBlob(
    slang::ICompileRequest* request,
    int translationUnitIndex,
    char const* path,
    ISlangBlob* sourceBlob)
{
    SLANG_ASSERT(request);
    request->addTranslationUnitSourceBlob(translationUnitIndex, path, sourceBlob);
}

SLANG_API int spAddEntryPoint(
    slang::ICompileRequest* request,
    int translationUnitIndex,
    char const* name,
    SlangStage stage)
{
    SLANG_ASSERT(request);
    return request->addEntryPoint(translationUnitIndex, name, stage);
}

SLANG_API int spAddEntryPointEx(
    slang::ICompileRequest* request,
    int translationUnitIndex,
    char const* name,
    SlangStage stage,
    int genericParamTypeNameCount,
    char const** genericParamTypeNames)
{
    SLANG_ASSERT(request);
    return request->addEntryPointEx(
        translationUnitIndex,
        name,
        stage,
        genericParamTypeNameCount,
        genericParamTypeNames);
}

SLANG_API SlangResult spSetGlobalGenericArgs(
    slang::ICompileRequest* request,
    int genericArgCount,
    char const** genericArgs)
{
    SLANG_ASSERT(request);
    return request->setGlobalGenericArgs(genericArgCount, genericArgs);
}

SLANG_API SlangResult spSetTypeNameForGlobalExistentialTypeParam(
    slang::ICompileRequest* request,
    int slotIndex,
    char const* typeName)
{
    SLANG_ASSERT(request);
    return request->setTypeNameForGlobalExistentialTypeParam(slotIndex, typeName);
}

SLANG_API SlangResult spSetTypeNameForEntryPointExistentialTypeParam(
    slang::ICompileRequest* request,
    int entryPointIndex,
    int slotIndex,
    char const* typeName)
{
    SLANG_ASSERT(request);
    return request->setTypeNameForEntryPointExistentialTypeParam(
        entryPointIndex,
        slotIndex,
        typeName);
}

SLANG_API SlangResult spCompile(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    return request->compile();
}

SLANG_API int spGetDependencyFileCount(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    return request->getDependencyFileCount();
}

SLANG_API char const* spGetDependencyFilePath(slang::ICompileRequest* request, int index)
{
    SLANG_ASSERT(request);
    return request->getDependencyFilePath(index);
}

SLANG_API int spGetTranslationUnitCount(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    return request->getTranslationUnitCount();
}

SLANG_API void const* spGetEntryPointCode(
    slang::ICompileRequest* request,
    int entryPointIndex,
    size_t* outSize)
{
    SLANG_ASSERT(request);
    return request->getEntryPointCode(entryPointIndex, outSize);
}

SLANG_API SlangResult spGetEntryPointCodeBlob(
    slang::ICompileRequest* request,
    int entryPointIndex,
    int targetIndex,
    ISlangBlob** outBlob)
{
    SLANG_ASSERT(request);
    return request->getEntryPointCodeBlob(entryPointIndex, targetIndex, outBlob);
}

SLANG_API SlangResult spGetEntryPointHostCallable(
    slang::ICompileRequest* request,
    int entryPointIndex,
    int targetIndex,
    ISlangSharedLibrary** outSharedLibrary)
{
    SLANG_ASSERT(request);
    return request->getEntryPointHostCallable(entryPointIndex, targetIndex, outSharedLibrary);
}

SLANG_API SlangResult
spGetTargetCodeBlob(slang::ICompileRequest* request, int targetIndex, ISlangBlob** outBlob)
{
    SLANG_ASSERT(request);
    return request->getTargetCodeBlob(targetIndex, outBlob);
}

SLANG_API SlangResult spGetTargetHostCallable(
    slang::ICompileRequest* request,
    int targetIndex,
    ISlangSharedLibrary** outSharedLibrary)
{
    SLANG_ASSERT(request);
    return request->getTargetHostCallable(targetIndex, outSharedLibrary);
}

SLANG_API char const* spGetEntryPointSource(slang::ICompileRequest* request, int entryPointIndex)
{
    SLANG_ASSERT(request);
    return request->getEntryPointSource(entryPointIndex);
}

SLANG_API void const* spGetCompileRequestCode(slang::ICompileRequest* request, size_t* outSize)
{
    SLANG_ASSERT(request);
    return request->getCompileRequestCode(outSize);
}

SLANG_API SlangResult spGetContainerCode(slang::ICompileRequest* request, ISlangBlob** outBlob)
{
    SLANG_ASSERT(request);
    return request->getContainerCode(outBlob);
}

SLANG_API SlangResult spLoadRepro(
    slang::ICompileRequest* request,
    ISlangFileSystem* fileSystem,
    const void* data,
    size_t size)
{
    SLANG_ASSERT(request);
    return request->loadRepro(fileSystem, data, size);
}

SLANG_API SlangResult spSaveRepro(slang::ICompileRequest* request, ISlangBlob** outBlob)
{
    SLANG_ASSERT(request);
    return request->saveRepro(outBlob);
}

SLANG_API SlangResult spEnableReproCapture(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    return request->enableReproCapture();
}

SLANG_API SlangResult
spCompileRequest_getProgram(slang::ICompileRequest* request, slang::IComponentType** outProgram)
{
    SLANG_ASSERT(request);
    return request->getProgram(outProgram);
}

SLANG_API SlangResult spCompileRequest_getProgramWithEntryPoints(
    slang::ICompileRequest* request,
    slang::IComponentType** outProgram)
{
    SLANG_ASSERT(request);
    return request->getProgramWithEntryPoints(outProgram);
}

SLANG_API SlangResult spCompileRequest_getModule(
    slang::ICompileRequest* request,
    SlangInt translationUnitIndex,
    slang::IModule** outModule)
{
    SLANG_ASSERT(request);
    return request->getModule(translationUnitIndex, outModule);
}

SLANG_API SlangResult
spCompileRequest_getSession(slang::ICompileRequest* request, slang::ISession** outSession)
{
    SLANG_ASSERT(request);
    return request->getSession(outSession);
}

SLANG_API SlangResult spCompileRequest_getEntryPoint(
    slang::ICompileRequest* request,
    SlangInt entryPointIndex,
    slang::IComponentType** outEntryPoint)
{
    SLANG_ASSERT(request);
    return request->getEntryPoint(entryPointIndex, outEntryPoint);
}

/*! @see slang::ICompileRequest::getCompileTimeProfile */
SLANG_API SlangResult spGetCompileTimeProfile(
    slang::ICompileRequest* request,
    ISlangProfiler** compileTimeProfile,
    bool shouldClear)
{
    SLANG_ASSERT(request);
    return request->getCompileTimeProfile(compileTimeProfile, shouldClear);
}

// Get the output code associated with a specific translation unit
SLANG_API char const* spGetTranslationUnitSource(
    slang::ICompileRequest* /*request*/,
    int /*translationUnitIndex*/
)
{
    fprintf(stderr, "DEPRECATED: spGetTranslationUnitSource()\n");
    return nullptr;
}

SLANG_API SlangResult
spProcessCommandLineArguments(SlangCompileRequest* request, char const* const* args, int argCount)
{
    return request->processCommandLineArguments(args, argCount);
}

// Reflection API

SLANG_API SlangReflection* spGetReflection(slang::ICompileRequest* request)
{
    SLANG_ASSERT(request);
    return request->getReflection();
}

// ... rest of reflection API implementation is in `Reflection.cpp`

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! Session !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

SLANG_API SlangResult spExtractRepro(
    SlangSession* session,
    const void* reproData,
    size_t reproDataSize,
    ISlangMutableFileSystem* fileSystem)
{
    using namespace Slang;
    SLANG_UNUSED(session);

    DiagnosticSink sink;
    sink.init(nullptr, nullptr);

    List<uint8_t> buffer;
    {
        MemoryStreamBase memoryStream(FileAccess::Read, reproData, reproDataSize);
        SLANG_RETURN_ON_FAIL(ReproUtil::loadState(&memoryStream, &sink, buffer));
    }

    MemoryOffsetBase base;
    base.set(buffer.getBuffer(), buffer.getCount());

    ReproUtil::RequestState* requestState = ReproUtil::getRequest(buffer);
    return ReproUtil::extractFiles(base, requestState, fileSystem);
}

SLANG_API SlangResult spLoadReproAsFileSystem(
    SlangSession* session,
    const void* reproData,
    size_t reproDataSize,
    ISlangFileSystem* replaceFileSystem,
    ISlangFileSystemExt** outFileSystem)
{
    using namespace Slang;

    SLANG_UNUSED(session);

    DiagnosticSink sink;
    sink.init(nullptr, nullptr);

    MemoryStreamBase stream(FileAccess::Read, reproData, reproDataSize);

    List<uint8_t> buffer;
    SLANG_RETURN_ON_FAIL(ReproUtil::loadState(&stream, &sink, buffer));

    auto requestState = ReproUtil::getRequest(buffer);
    MemoryOffsetBase base;
    base.set(buffer.getBuffer(), buffer.getCount());

    ComPtr<ISlangFileSystemExt> fileSystem;
    SLANG_RETURN_ON_FAIL(
        ReproUtil::loadFileSystem(base, requestState, replaceFileSystem, fileSystem));

    *outFileSystem = fileSystem.detach();
    return SLANG_OK;
}

SLANG_API void spOverrideDiagnosticSeverity(
    slang::ICompileRequest* request,
    SlangInt messageID,
    SlangSeverity overrideSeverity)
{
    if (!request)
        return;

    request->overrideDiagnosticSeverity(messageID, overrideSeverity);
}

SLANG_API SlangDiagnosticFlags spGetDiagnosticFlags(slang::ICompileRequest* request)
{
    if (!request)
        return 0;

    return request->getDiagnosticFlags();
}

SLANG_API void spSetDiagnosticFlags(slang::ICompileRequest* request, SlangDiagnosticFlags flags)
{
    if (!request)
        return;

    request->setDiagnosticFlags(flags);
}
