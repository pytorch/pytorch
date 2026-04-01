#ifndef JSON_CONSUMER_H
#define JSON_CONSUMER_H

#include "../../core/slang-stream.h"
#include "../util/record-format.h"
#include "../util/record-utility.h"
#include "decoder-consumer.h"

namespace SlangRecord
{
class CommonInterfaceWriter
{
public:
    CommonInterfaceWriter(ApiClassId classId, Slang::FileStream& fileStream)
        : m_fileStream(fileStream)
    {
        switch (classId)
        {
        case ApiClassId::Class_IModule:
            m_className = "IModule";
            break;
        case ApiClassId::Class_IEntryPoint:
            m_className = "IEntryPoint";
            break;
        case ApiClassId::Class_ICompositeComponentType:
            m_className = "ICompositeComponentType";
            break;
        case ApiClassId::Class_ITypeConformance:
            m_className = "ITypeConformance";
            break;
        default:
            slangRecordLog(LogLevel::Error, "Invalid classNo %u\n", classId);
            break;
        }
    }
    CommonInterfaceWriter() = delete;
    void getSession(ObjectID objectId, ObjectID outSessionId);
    void getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId);
    void getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    void getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    void getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystemId);
    void getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId);
    void specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId);
    void link(ObjectID objectId, ObjectID outLinkedComponentTypeId, ObjectID outDiagnosticsId);
    void getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibraryId,
        ObjectID outDiagnosticsId);
    void renameEntryPoint(ObjectID objectId, const char* newName, ObjectID outEntryPointId);
    void linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId);

protected:
    Slang::String m_className;
    Slang::FileStream& m_fileStream;
};

class JsonConsumer : public IDecoderConsumer, public Slang::RefObject
{
public:
    JsonConsumer(const Slang::String& filePath);
    virtual ~JsonConsumer() = default;
    virtual void CreateGlobalSession(
        SlangGlobalSessionDesc const& desc,
        ObjectID outGlobalSessionId);
    virtual void IGlobalSession_createSession(
        ObjectID objectId,
        slang::SessionDesc const& desc,
        ObjectID outSessionId);
    virtual void IGlobalSession_findProfile(ObjectID objectId, char const* name);
    virtual void IGlobalSession_setDownstreamCompilerPath(
        ObjectID objectId,
        SlangPassThrough passThrough,
        char const* path);
    virtual void IGlobalSession_setDownstreamCompilerPrelude(
        ObjectID objectId,
        SlangPassThrough inPassThrough,
        char const* prelude);
    virtual void IGlobalSession_getDownstreamCompilerPrelude(
        ObjectID objectId,
        SlangPassThrough inPassThrough,
        ObjectID outPreludeId);

    virtual void IGlobalSession_getBuildTagString(ObjectID objectId) { (void)objectId; }

    virtual void IGlobalSession_setDefaultDownstreamCompiler(
        ObjectID objectId,
        SlangSourceLanguage sourceLanguage,
        SlangPassThrough defaultCompiler);
    virtual void IGlobalSession_getDefaultDownstreamCompiler(
        ObjectID objectId,
        SlangSourceLanguage sourceLanguage);
    virtual void IGlobalSession_setLanguagePrelude(
        ObjectID objectId,
        SlangSourceLanguage inSourceLanguage,
        char const* prelude);
    virtual void IGlobalSession_getLanguagePrelude(
        ObjectID objectId,
        SlangSourceLanguage inSourceLanguage,
        ObjectID outPreludeId);
    virtual void IGlobalSession_createCompileRequest(ObjectID objectId, ObjectID outCompileRequest);
    virtual void IGlobalSession_addBuiltins(
        ObjectID objectId,
        char const* sourcePath,
        char const* sourceString);
    virtual void IGlobalSession_setSharedLibraryLoader(ObjectID objectId, ObjectID loaderId);
    virtual void IGlobalSession_getSharedLibraryLoader(ObjectID objectId, ObjectID outLoaderId);
    virtual void IGlobalSession_checkCompileTargetSupport(
        ObjectID objectId,
        SlangCompileTarget target);
    virtual void IGlobalSession_checkPassThroughSupport(
        ObjectID objectId,
        SlangPassThrough passThrough);
    virtual void IGlobalSession_compileCoreModule(
        ObjectID objectId,
        slang::CompileCoreModuleFlags flags);
    virtual void IGlobalSession_loadCoreModule(
        ObjectID objectId,
        const void* coreModule,
        size_t coreModuleSizeInBytes);
    virtual void IGlobalSession_saveCoreModule(
        ObjectID objectId,
        SlangArchiveType archiveType,
        ObjectID outBlobId);
    virtual void IGlobalSession_findCapability(ObjectID objectId, char const* name);
    virtual void IGlobalSession_setDownstreamCompilerForTransition(
        ObjectID objectId,
        SlangCompileTarget source,
        SlangCompileTarget target,
        SlangPassThrough compiler);
    virtual void IGlobalSession_getDownstreamCompilerForTransition(
        ObjectID objectId,
        SlangCompileTarget source,
        SlangCompileTarget target);

    virtual void IGlobalSession_getCompilerElapsedTime(ObjectID objectId) { (void)objectId; }

    virtual void IGlobalSession_setSPIRVCoreGrammar(ObjectID objectId, char const* jsonPath);
    virtual void IGlobalSession_parseCommandLineArguments(
        ObjectID objectId,
        int argc,
        const char* const* argv,
        ObjectID outSessionDescId,
        ObjectID outAllocationId);
    virtual void IGlobalSession_getSessionDescDigest(
        ObjectID objectId,
        slang::SessionDesc* sessionDesc,
        ObjectID outBlobId);

    // ISession
    virtual void ISession_getGlobalSession(ObjectID objectId, ObjectID outGlobalSessionId);
    virtual void ISession_loadModule(
        ObjectID objectId,
        const char* moduleName,
        ObjectID outDiagnostics,
        ObjectID outModuleId);

    virtual void ISession_loadModuleFromIRBlob(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId);
    virtual void ISession_loadModuleFromSource(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId);
    virtual void ISession_loadModuleFromSourceString(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        const char* string,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId);
    virtual void ISession_createCompositeComponentType(
        ObjectID objectId,
        ObjectID* componentTypeIds,
        SlangInt componentTypeCount,
        ObjectID outCompositeComponentTypeIds,
        ObjectID outDiagnosticsId);

    virtual void ISession_specializeType(
        ObjectID objectId,
        ObjectID typeId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outDiagnosticsId,
        ObjectID outTypeReflectionId);

    virtual void ISession_getTypeLayout(
        ObjectID objectId,
        ObjectID typeId,
        SlangInt targetIndex,
        slang::LayoutRules rules,
        ObjectID outDiagnosticsId,
        ObjectID outTypeLayoutReflection);

    virtual void ISession_getContainerType(
        ObjectID objectId,
        ObjectID elementType,
        slang::ContainerType containerType,
        ObjectID outDiagnosticsId,
        ObjectID outTypeReflectionId);

    virtual void ISession_getDynamicType(ObjectID objectId, ObjectID outTypeReflectionId);

    virtual void ISession_getTypeRTTIMangledName(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID outNameBlobId);

    virtual void ISession_getTypeConformanceWitnessMangledName(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        ObjectID outNameBlobId);

    virtual void ISession_getTypeConformanceWitnessSequentialID(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        uint32_t outId);

    virtual void ISession_createTypeConformanceComponentType(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        ObjectID outConformanceId,
        SlangInt conformanceIdOverride,
        ObjectID outDiagnosticsId);

    virtual void ISession_createCompileRequest(ObjectID objectId, ObjectID outCompileRequestId);

    virtual void ISession_getLoadedModuleCount(ObjectID objectId) { (void)objectId; }

    virtual void ISession_getLoadedModule(ObjectID objectId, SlangInt index, ObjectID outModuleId);

    virtual void ISession_isBinaryModuleUpToDate(ObjectID objectId) { (void)objectId; }

    // IModule
    virtual void IModule_findEntryPointByName(
        ObjectID objectId,
        char const* name,
        ObjectID outEntryPointId);

    virtual void IModule_getDefinedEntryPointCount(ObjectID objectId) { (void)objectId; }

    virtual void IModule_getDefinedEntryPoint(
        ObjectID objectId,
        SlangInt32 index,
        ObjectID outEntryPointId);
    virtual void IModule_serialize(ObjectID objectId, ObjectID outSerializedBlobId);
    virtual void IModule_writeToFile(ObjectID objectId, char const* fileName);

    virtual void IModule_getName(ObjectID objectId) { (void)objectId; }
    virtual void IModule_getFilePath(ObjectID objectId) { (void)objectId; }
    virtual void IModule_getUniqueIdentity(ObjectID objectId) { (void)objectId; }

    virtual void IModule_findAndCheckEntryPoint(
        ObjectID objectId,
        char const* name,
        SlangStage stage,
        ObjectID outEntryPointId,
        ObjectID outDiagnostics);

    virtual void IModule_getSession(ObjectID objectId, ObjectID outSessionId);
    virtual void IModule_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId);

    virtual void IModule_getSpecializationParamCount(ObjectID objectId) { (void)objectId; }

    virtual void IModule_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    virtual void IModule_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    virtual void IModule_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem);
    virtual void IModule_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId);
    virtual void IModule_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void IModule_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void IModule_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics);
    virtual void IModule_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId);
    virtual void IModule_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId);

    // IEntryPoint
    virtual void IEntryPoint_getSession(ObjectID objectId, ObjectID outSessionId);
    virtual void IEntryPoint_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId);

    virtual void IEntryPoint_getSpecializationParamCount(ObjectID objectId) { (void)objectId; };

    virtual void IEntryPoint_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    virtual void IEntryPoint_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    virtual void IEntryPoint_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem);
    virtual void IEntryPoint_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId);
    virtual void IEntryPoint_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void IEntryPoint_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void IEntryPoint_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics);
    virtual void IEntryPoint_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId);
    virtual void IEntryPoint_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId);

    // ICompositeComponentType
    virtual void ICompositeComponentType_getSession(ObjectID objectId, ObjectID outSessionId);
    virtual void ICompositeComponentType_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId);

    virtual void ICompositeComponentType_getSpecializationParamCount(ObjectID objectId)
    {
        (void)objectId;
    };

    virtual void ICompositeComponentType_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnostics);
    virtual void ICompositeComponentType_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnostics);
    virtual void ICompositeComponentType_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem);
    virtual void ICompositeComponentType_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId);
    virtual void ICompositeComponentType_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void ICompositeComponentType_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void ICompositeComponentType_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics);
    virtual void ICompositeComponentType_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId);
    virtual void ICompositeComponentType_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId);

    // ITypeConformance
    virtual void ITypeConformance_getSession(ObjectID objectId, ObjectID outSessionId);
    virtual void ITypeConformance_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId);

    virtual void ITypeConformance_getSpecializationParamCount(ObjectID objectId)
    {
        (void)objectId;
    };

    virtual void ITypeConformance_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    virtual void ITypeConformance_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId);
    virtual void ITypeConformance_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem);
    virtual void ITypeConformance_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId);
    virtual void ITypeConformance_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void ITypeConformance_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId);
    virtual void ITypeConformance_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics);
    virtual void ITypeConformance_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId);
    virtual void ITypeConformance_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId);

    static void _writeCompilerOptionEntryHelper(
        Slang::StringBuilder& builder,
        int indent,
        slang::CompilerOptionEntry* compilerOptionEntries,
        uint32_t compilerOptionEntryCount,
        bool isLast = false);
    static void _writeGlobalSessionDescHelper(
        Slang::StringBuilder& builder,
        int indent,
        SlangGlobalSessionDesc const& globalSessionDesc,
        Slang::String keyName,
        bool isLast = false);
    static void _writeSessionDescHelper(
        Slang::StringBuilder& builder,
        int indent,
        slang::SessionDesc const& sessionDesc,
        Slang::String keyName,
        bool isLast = false);

private:
    Slang::FileStream m_fileStream;
    bool m_isFileValid = false;
    CommonInterfaceWriter m_moduleHelper{ApiClassId::Class_IModule, m_fileStream};
    CommonInterfaceWriter m_entryPointHelper{ApiClassId::Class_IEntryPoint, m_fileStream};
    CommonInterfaceWriter m_compositeComponentTypeHelper{
        ApiClassId::Class_ICompositeComponentType,
        m_fileStream};
    CommonInterfaceWriter m_typeConformanceHelper{ApiClassId::Class_ITypeConformance, m_fileStream};
};
} // namespace SlangRecord
#endif // JSON_CONSUMER_H
