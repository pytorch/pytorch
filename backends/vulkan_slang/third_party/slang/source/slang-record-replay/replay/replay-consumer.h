#ifndef REPLAY_CONSUMER_H
#define REPLAY_CONSUMER_H

#include "../../core/slang-stream.h"
#include "../util/record-format.h"
#include "../util/record-utility.h"
#include "decoder-consumer.h"

#include <unordered_map>

namespace SlangRecord
{
// class CommonInterfaceReplayer;

class CommonInterfaceReplayer
{
public:
    CommonInterfaceReplayer(Slang::Dictionary<ObjectID, void*>& pObjectMap)
        : m_objectMap(pObjectMap)
    {
    }
    virtual ~CommonInterfaceReplayer() = default;

    SlangResult getSession(ObjectID objectId, ObjectID outSessionId) { return SLANG_FAIL; }

    SlangResult getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId);
    SlangResult getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    SlangResult getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    SlangResult getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystemId);
    SlangResult getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId);
    SlangResult specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId);
    SlangResult link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId);
    SlangResult getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibraryId,
        ObjectID outDiagnosticsId);
    SlangResult renameEntryPoint(ObjectID objectId, const char* newName, ObjectID outEntryPointId);
    SlangResult linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId);

private:
    inline slang::IComponentType* getObjectPointer(ObjectID objectId)
    {
        void* objPtr = nullptr;

        // If the object is not found, there must be something wrong with the record/replay
        // logic, so report an error.
        if (!m_objectMap.tryGetValue(objectId, objPtr))
        {
            slangRecordLog(LogLevel::Error, "Object not found in the object map: %d\n", objectId);
            std::abort();
        }

        return static_cast<slang::IComponentType*>(objPtr);
    }

    Slang::Dictionary<ObjectID, void*>& m_objectMap;
    uint32_t m_globalCounter = 0;
};

class ReplayConsumer : public IDecoderConsumer, public Slang::RefObject
{
public:
    virtual void CreateGlobalSession(
        SlangGlobalSessionDesc const& desc,
        ObjectID outGlobalSessionId) override;
    virtual void IGlobalSession_createSession(
        ObjectID objectId,
        slang::SessionDesc const& desc,
        ObjectID outSessionId) override;
    virtual void IGlobalSession_findProfile(ObjectID objectId, char const* name) override;
    virtual void IGlobalSession_setDownstreamCompilerPath(
        ObjectID objectId,
        SlangPassThrough passThrough,
        char const* path) override;
    virtual void IGlobalSession_setDownstreamCompilerPrelude(
        ObjectID objectId,
        SlangPassThrough inPassThrough,
        char const* prelude) override;
    virtual void IGlobalSession_getDownstreamCompilerPrelude(
        ObjectID objectId,
        SlangPassThrough inPassThrough,
        ObjectID outPreludeId) override;

    virtual void IGlobalSession_getBuildTagString(ObjectID objectId) override { (void)objectId; }

    virtual void IGlobalSession_setDefaultDownstreamCompiler(
        ObjectID objectId,
        SlangSourceLanguage sourceLanguage,
        SlangPassThrough defaultCompiler) override;
    virtual void IGlobalSession_getDefaultDownstreamCompiler(
        ObjectID objectId,
        SlangSourceLanguage sourceLanguage) override;
    virtual void IGlobalSession_setLanguagePrelude(
        ObjectID objectId,
        SlangSourceLanguage inSourceLanguage,
        char const* prelude) override;
    virtual void IGlobalSession_getLanguagePrelude(
        ObjectID objectId,
        SlangSourceLanguage inSourceLanguage,
        ObjectID outPreludeId) override;
    virtual void IGlobalSession_createCompileRequest(ObjectID objectId, ObjectID outCompileRequest)
        override;
    virtual void IGlobalSession_addBuiltins(
        ObjectID objectId,
        char const* sourcePath,
        char const* sourceString) override;
    virtual void IGlobalSession_setSharedLibraryLoader(ObjectID objectId, ObjectID loaderId)
        override;
    virtual void IGlobalSession_getSharedLibraryLoader(ObjectID objectId, ObjectID outLoaderId)
        override;
    virtual void IGlobalSession_checkCompileTargetSupport(
        ObjectID objectId,
        SlangCompileTarget target) override;
    virtual void IGlobalSession_checkPassThroughSupport(
        ObjectID objectId,
        SlangPassThrough passThrough) override;
    virtual void IGlobalSession_compileCoreModule(
        ObjectID objectId,
        slang::CompileCoreModuleFlags flags) override;
    virtual void IGlobalSession_loadCoreModule(
        ObjectID objectId,
        const void* coreModule,
        size_t coreModuleSizeInBytes) override;
    virtual void IGlobalSession_saveCoreModule(
        ObjectID objectId,
        SlangArchiveType archiveType,
        ObjectID outBlobId) override;
    virtual void IGlobalSession_findCapability(ObjectID objectId, char const* name) override;
    virtual void IGlobalSession_setDownstreamCompilerForTransition(
        ObjectID objectId,
        SlangCompileTarget source,
        SlangCompileTarget target,
        SlangPassThrough compiler) override;
    virtual void IGlobalSession_getDownstreamCompilerForTransition(
        ObjectID objectId,
        SlangCompileTarget source,
        SlangCompileTarget target) override;

    virtual void IGlobalSession_getCompilerElapsedTime(ObjectID objectId) override
    {
        (void)objectId;
    }

    virtual void IGlobalSession_setSPIRVCoreGrammar(ObjectID objectId, char const* jsonPath)
        override;
    virtual void IGlobalSession_parseCommandLineArguments(
        ObjectID objectId,
        int argc,
        const char* const* argv,
        ObjectID outSessionDescId,
        ObjectID outAllocationId) override;
    virtual void IGlobalSession_getSessionDescDigest(
        ObjectID objectId,
        slang::SessionDesc* sessionDesc,
        ObjectID outBlobId) override;

    // ISession
    virtual void ISession_getGlobalSession(ObjectID objectId, ObjectID outGlobalSessionId) override;
    virtual void ISession_loadModule(
        ObjectID objectId,
        const char* moduleName,
        ObjectID outDiagnostics,
        ObjectID outModuleId) override;

    virtual void ISession_loadModuleFromIRBlob(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId) override;
    virtual void ISession_loadModuleFromSource(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId) override;
    virtual void ISession_loadModuleFromSourceString(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        const char* string,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId) override;
    virtual void ISession_createCompositeComponentType(
        ObjectID objectId,
        ObjectID* componentTypeIds,
        SlangInt componentTypeCount,
        ObjectID outCompositeComponentTypeIds,
        ObjectID outDiagnosticsId) override;

    virtual void ISession_specializeType(
        ObjectID objectId,
        ObjectID typeId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outDiagnosticsId,
        ObjectID outTypeReflectionId) override;

    virtual void ISession_getTypeLayout(
        ObjectID objectId,
        ObjectID typeId,
        SlangInt targetIndex,
        slang::LayoutRules rules,
        ObjectID outDiagnosticsId,
        ObjectID outTypeLayoutReflection) override;

    virtual void ISession_getContainerType(
        ObjectID objectId,
        ObjectID elementType,
        slang::ContainerType containerType,
        ObjectID outDiagnosticsId,
        ObjectID outTypeReflectionId) override;

    virtual void ISession_getDynamicType(ObjectID objectId, ObjectID outTypeReflectionId) override;

    virtual void ISession_getTypeRTTIMangledName(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID outNameBlobId) override;

    virtual void ISession_getTypeConformanceWitnessMangledName(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        ObjectID outNameBlobId) override;

    virtual void ISession_getTypeConformanceWitnessSequentialID(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        uint32_t outId) override;

    virtual void ISession_createTypeConformanceComponentType(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        ObjectID outConformanceId,
        SlangInt conformanceIdOverride,
        ObjectID outDiagnosticsId) override;

    virtual void ISession_createCompileRequest(ObjectID objectId, ObjectID outCompileRequestId)
        override;

    virtual void ISession_getLoadedModuleCount(ObjectID objectId) override { (void)objectId; }

    virtual void ISession_getLoadedModule(ObjectID objectId, SlangInt index, ObjectID outModuleId)
        override;

    virtual void ISession_isBinaryModuleUpToDate(ObjectID objectId) override { (void)objectId; }

    // IModule
    virtual void IModule_findEntryPointByName(
        ObjectID objectId,
        char const* name,
        ObjectID outEntryPointId) override;

    virtual void IModule_getDefinedEntryPointCount(ObjectID objectId) override { (void)objectId; }

    virtual void IModule_getDefinedEntryPoint(
        ObjectID objectId,
        SlangInt32 index,
        ObjectID outEntryPointId) override;
    virtual void IModule_serialize(ObjectID objectId, ObjectID outSerializedBlobId) override;
    virtual void IModule_writeToFile(ObjectID objectId, char const* fileName) override;

    virtual void IModule_getName(ObjectID objectId) override { (void)objectId; }
    virtual void IModule_getFilePath(ObjectID objectId) override { (void)objectId; }
    virtual void IModule_getUniqueIdentity(ObjectID objectId) override { (void)objectId; }

    virtual void IModule_findAndCheckEntryPoint(
        ObjectID objectId,
        char const* name,
        SlangStage stage,
        ObjectID outEntryPointId,
        ObjectID outDiagnostics) override;

    virtual void IModule_getSession(ObjectID objectId, ObjectID outSessionId) override;
    virtual void IModule_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) override;

    virtual void IModule_getSpecializationParamCount(ObjectID objectId) override { (void)objectId; }

    virtual void IModule_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void IModule_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void IModule_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) override;
    virtual void IModule_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) override;
    virtual void IModule_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void IModule_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void IModule_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) override;
    virtual void IModule_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) override;
    virtual void IModule_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) override;

    // IEntryPoint
    virtual void IEntryPoint_getSession(ObjectID objectId, ObjectID outSessionId) override;
    virtual void IEntryPoint_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) override;

    virtual void IEntryPoint_getSpecializationParamCount(ObjectID objectId) override
    {
        (void)objectId;
    };

    virtual void IEntryPoint_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void IEntryPoint_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void IEntryPoint_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) override;
    virtual void IEntryPoint_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) override;
    virtual void IEntryPoint_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void IEntryPoint_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void IEntryPoint_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) override;
    virtual void IEntryPoint_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) override;
    virtual void IEntryPoint_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) override;

    // ICompositeComponentType
    virtual void ICompositeComponentType_getSession(ObjectID objectId, ObjectID outSessionId)
        override;
    virtual void ICompositeComponentType_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) override;

    virtual void ICompositeComponentType_getSpecializationParamCount(ObjectID objectId) override
    {
        (void)objectId;
    };

    virtual void ICompositeComponentType_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void ICompositeComponentType_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void ICompositeComponentType_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) override;
    virtual void ICompositeComponentType_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) override;
    virtual void ICompositeComponentType_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void ICompositeComponentType_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void ICompositeComponentType_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) override;
    virtual void ICompositeComponentType_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) override;
    virtual void ICompositeComponentType_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) override;

    // ITypeConformance
    virtual void ITypeConformance_getSession(ObjectID objectId, ObjectID outSessionId) override;
    virtual void ITypeConformance_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) override;

    virtual void ITypeConformance_getSpecializationParamCount(ObjectID objectId) override
    {
        (void)objectId;
    };

    virtual void ITypeConformance_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void ITypeConformance_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) override;
    virtual void ITypeConformance_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) override;
    virtual void ITypeConformance_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) override;
    virtual void ITypeConformance_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void ITypeConformance_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) override;
    virtual void ITypeConformance_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) override;
    virtual void ITypeConformance_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) override;
    virtual void ITypeConformance_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) override;

    static void printDiagnosticMessage(slang::IBlob* diagnosticsBlob);

private:
    // Map of the address of the object allocated by slang during record to
    // the address of the object allocated by the replay.
    // We need to have this map because we never save the content of the object
    // allocated by slang. Because those are just opaque objects or handles, we
    // only need to provide them to the corresponding replay function or call the
    // methods on the correct object.
    Slang::Dictionary<ObjectID, void*> m_objectMap;

    template<typename T>
    inline T* getObjectPointer(ObjectID objectId)
    {
        void* objPtr = nullptr;
        if (!m_objectMap.tryGetValue(objectId, objPtr))
        {
            slangRecordLog(LogLevel::Error, "Object not found in the object map: %d\n", objectId);
            std::abort();
        }

        return static_cast<T*>(objPtr);
    }

    CommonInterfaceReplayer m_commonReplayer{m_objectMap};
};
} // namespace SlangRecord
#endif // REPLAY_CONSUMER_H
