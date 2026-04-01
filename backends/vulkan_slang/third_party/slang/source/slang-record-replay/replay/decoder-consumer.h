#ifndef DECODER_CONSUMER_H
#define DECODER_CONSUMER_H

#include "../../core/slang-stream.h"
#include "../util/record-format.h"
#include "../util/record-utility.h"
#include "slang.h"

namespace SlangRecord
{
class IDecoderConsumer
{
public:
    virtual void CreateGlobalSession(
        SlangGlobalSessionDesc const& desc,
        ObjectID outGlobalSessionId) = 0;
    virtual void IGlobalSession_createSession(
        ObjectID objectId,
        slang::SessionDesc const& desc,
        ObjectID outSessionId) = 0;
    virtual void IGlobalSession_findProfile(ObjectID objectId, char const* name) = 0;
    virtual void IGlobalSession_setDownstreamCompilerPath(
        ObjectID objectId,
        SlangPassThrough passThrough,
        char const* path) = 0;
    virtual void IGlobalSession_setDownstreamCompilerPrelude(
        ObjectID objectId,
        SlangPassThrough inPassThrough,
        char const* prelude) = 0;
    virtual void IGlobalSession_getDownstreamCompilerPrelude(
        ObjectID objectId,
        SlangPassThrough inPassThrough,
        ObjectID outPreludeId) = 0;

    virtual void IGlobalSession_getBuildTagString(ObjectID objectId) { (void)objectId; }

    virtual void IGlobalSession_setDefaultDownstreamCompiler(
        ObjectID objectId,
        SlangSourceLanguage sourceLanguage,
        SlangPassThrough defaultCompiler) = 0;
    virtual void IGlobalSession_getDefaultDownstreamCompiler(
        ObjectID objectId,
        SlangSourceLanguage sourceLanguage) = 0;
    virtual void IGlobalSession_setLanguagePrelude(
        ObjectID objectId,
        SlangSourceLanguage inSourceLanguage,
        char const* prelude) = 0;
    virtual void IGlobalSession_getLanguagePrelude(
        ObjectID objectId,
        SlangSourceLanguage inSourceLanguage,
        ObjectID outPreludeId) = 0;
    virtual void IGlobalSession_createCompileRequest(
        ObjectID objectId,
        ObjectID outCompileRequest) = 0;
    virtual void IGlobalSession_addBuiltins(
        ObjectID objectId,
        char const* sourcePath,
        char const* sourceString) = 0;
    virtual void IGlobalSession_setSharedLibraryLoader(ObjectID objectId, ObjectID loaderId) = 0;
    virtual void IGlobalSession_getSharedLibraryLoader(ObjectID objectId, ObjectID outLoaderId) = 0;
    virtual void IGlobalSession_checkCompileTargetSupport(
        ObjectID objectId,
        SlangCompileTarget target) = 0;
    virtual void IGlobalSession_checkPassThroughSupport(
        ObjectID objectId,
        SlangPassThrough passThrough) = 0;
    virtual void IGlobalSession_compileCoreModule(
        ObjectID objectId,
        slang::CompileCoreModuleFlags flags) = 0;
    virtual void IGlobalSession_loadCoreModule(
        ObjectID objectId,
        const void* coreModule,
        size_t coreModuleSizeInBytes) = 0;
    virtual void IGlobalSession_saveCoreModule(
        ObjectID objectId,
        SlangArchiveType archiveType,
        ObjectID outBlobId) = 0;
    virtual void IGlobalSession_findCapability(ObjectID objectId, char const* name) = 0;
    virtual void IGlobalSession_setDownstreamCompilerForTransition(
        ObjectID objectId,
        SlangCompileTarget source,
        SlangCompileTarget target,
        SlangPassThrough compiler) = 0;
    virtual void IGlobalSession_getDownstreamCompilerForTransition(
        ObjectID objectId,
        SlangCompileTarget source,
        SlangCompileTarget target) = 0;

    virtual void IGlobalSession_getCompilerElapsedTime(ObjectID objectId) { (void)objectId; }

    virtual void IGlobalSession_setSPIRVCoreGrammar(ObjectID objectId, char const* jsonPath) = 0;
    virtual void IGlobalSession_parseCommandLineArguments(
        ObjectID objectId,
        int argc,
        const char* const* argv,
        ObjectID outSessionDescId,
        ObjectID outAllocationId) = 0;
    virtual void IGlobalSession_getSessionDescDigest(
        ObjectID objectId,
        slang::SessionDesc* sessionDesc,
        ObjectID outBlobId) = 0;

    // ISession
    virtual void ISession_getGlobalSession(ObjectID objectId, ObjectID outGlobalSessionId) = 0;
    virtual void ISession_loadModule(
        ObjectID objectId,
        const char* moduleName,
        ObjectID outDiagnostics,
        ObjectID outModuleId) = 0;

    virtual void ISession_loadModuleFromIRBlob(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId) = 0;
    virtual void ISession_loadModuleFromSource(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId) = 0;
    virtual void ISession_loadModuleFromSourceString(
        ObjectID objectId,
        const char* moduleName,
        const char* path,
        const char* string,
        ObjectID outDiagnosticsId,
        ObjectID outModuleId) = 0;
    virtual void ISession_createCompositeComponentType(
        ObjectID objectId,
        ObjectID* componentTypeIds,
        SlangInt componentTypeCount,
        ObjectID outCompositeComponentTypeIds,
        ObjectID outDiagnosticsId) = 0;

    virtual void ISession_specializeType(
        ObjectID objectId,
        ObjectID typeId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outDiagnosticsId,
        ObjectID outTypeReflectionId) = 0;

    virtual void ISession_getTypeLayout(
        ObjectID objectId,
        ObjectID typeId,
        SlangInt targetIndex,
        slang::LayoutRules rules,
        ObjectID outDiagnosticsId,
        ObjectID outTypeLayoutReflection) = 0;

    virtual void ISession_getContainerType(
        ObjectID objectId,
        ObjectID elementType,
        slang::ContainerType containerType,
        ObjectID outDiagnosticsId,
        ObjectID outTypeReflectionId) = 0;

    virtual void ISession_getDynamicType(ObjectID objectId, ObjectID outTypeReflectionId) = 0;

    virtual void ISession_getTypeRTTIMangledName(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID outNameBlobId) = 0;

    virtual void ISession_getTypeConformanceWitnessMangledName(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        ObjectID outNameBlobId) = 0;

    virtual void ISession_getTypeConformanceWitnessSequentialID(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        uint32_t outId) = 0;

    virtual void ISession_createTypeConformanceComponentType(
        ObjectID objectId,
        ObjectID typeId,
        ObjectID interfaceTypeId,
        ObjectID outConformanceId,
        SlangInt conformanceIdOverride,
        ObjectID outDiagnosticsId) = 0;

    virtual void ISession_createCompileRequest(ObjectID objectId, ObjectID outCompileRequestId) = 0;

    virtual void ISession_getLoadedModuleCount(ObjectID objectId) { (void)objectId; }

    virtual void ISession_getLoadedModule(
        ObjectID objectId,
        SlangInt index,
        ObjectID outModuleId) = 0;

    virtual void ISession_isBinaryModuleUpToDate(ObjectID objectId) { (void)objectId; }

    // IModule
    virtual void IModule_findEntryPointByName(
        ObjectID objectId,
        char const* name,
        ObjectID outEntryPointId) = 0;

    virtual void IModule_getDefinedEntryPointCount(ObjectID objectId) { (void)objectId; }

    virtual void IModule_getDefinedEntryPoint(
        ObjectID objectId,
        SlangInt32 index,
        ObjectID outEntryPointId) = 0;
    virtual void IModule_serialize(ObjectID objectId, ObjectID outSerializedBlobId) = 0;
    virtual void IModule_writeToFile(ObjectID objectId, char const* fileName) = 0;

    virtual void IModule_getName(ObjectID objectId) { (void)objectId; }
    virtual void IModule_getFilePath(ObjectID objectId) { (void)objectId; }
    virtual void IModule_getUniqueIdentity(ObjectID objectId) { (void)objectId; }

    virtual void IModule_findAndCheckEntryPoint(
        ObjectID objectId,
        char const* name,
        SlangStage stage,
        ObjectID outEntryPointId,
        ObjectID outDiagnostics) = 0;

    virtual void IModule_getSession(ObjectID objectId, ObjectID outSessionId) = 0;
    virtual void IModule_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) = 0;

    virtual void IModule_getSpecializationParamCount(ObjectID objectId) { (void)objectId; }

    virtual void IModule_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IModule_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IModule_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) = 0;
    virtual void IModule_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) = 0;
    virtual void IModule_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IModule_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IModule_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) = 0;
    virtual void IModule_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) = 0;
    virtual void IModule_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) = 0;

    // IEntryPoint
    virtual void IEntryPoint_getSession(ObjectID objectId, ObjectID outSessionId) = 0;
    virtual void IEntryPoint_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) = 0;

    virtual void IEntryPoint_getSpecializationParamCount(ObjectID objectId) { (void)objectId; };

    virtual void IEntryPoint_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IEntryPoint_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IEntryPoint_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) = 0;
    virtual void IEntryPoint_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) = 0;
    virtual void IEntryPoint_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IEntryPoint_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void IEntryPoint_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) = 0;
    virtual void IEntryPoint_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) = 0;
    virtual void IEntryPoint_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) = 0;

    // ICompositeComponentType
    virtual void ICompositeComponentType_getSession(ObjectID objectId, ObjectID outSessionId) = 0;
    virtual void ICompositeComponentType_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) = 0;

    virtual void ICompositeComponentType_getSpecializationParamCount(ObjectID objectId)
    {
        (void)objectId;
    };

    virtual void ICompositeComponentType_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ICompositeComponentType_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ICompositeComponentType_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) = 0;
    virtual void ICompositeComponentType_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) = 0;
    virtual void ICompositeComponentType_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ICompositeComponentType_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ICompositeComponentType_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) = 0;
    virtual void ICompositeComponentType_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) = 0;
    virtual void ICompositeComponentType_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) = 0;

    // ITypeConformance
    virtual void ITypeConformance_getSession(ObjectID objectId, ObjectID outSessionId) = 0;
    virtual void ITypeConformance_getLayout(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outDiagnosticsId,
        ObjectID retProgramLayoutId) = 0;

    virtual void ITypeConformance_getSpecializationParamCount(ObjectID objectId)
    {
        (void)objectId;
    };

    virtual void ITypeConformance_getEntryPointCode(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ITypeConformance_getTargetCode(
        ObjectID objectId,
        SlangInt targetIndex,
        ObjectID outCodeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ITypeConformance_getResultAsFileSystem(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outFileSystem) = 0;
    virtual void ITypeConformance_getEntryPointHash(
        ObjectID objectId,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ObjectID outHashId) = 0;
    virtual void ITypeConformance_specialize(
        ObjectID objectId,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ObjectID outSpecializedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ITypeConformance_link(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        ObjectID outDiagnosticsId) = 0;
    virtual void ITypeConformance_getEntryPointHostCallable(
        ObjectID objectId,
        int entryPointIndex,
        int targetIndex,
        ObjectID outSharedLibrary,
        ObjectID outDiagnostics) = 0;
    virtual void ITypeConformance_renameEntryPoint(
        ObjectID objectId,
        const char* newName,
        ObjectID outEntryPointId) = 0;
    virtual void ITypeConformance_linkWithOptions(
        ObjectID objectId,
        ObjectID outLinkedComponentTypeId,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ObjectID outDiagnosticsId) = 0;
};
} // namespace SlangRecord


#endif // DECODER_CONSUMER_H
