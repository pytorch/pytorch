#ifndef SLANG_DECODER_H
#define SLANG_DECODER_H

#include "../../core/slang-list.h"
#include "../util/record-format.h"
#include "decoder-consumer.h"

#include <unordered_map>
#include <vector>

namespace SlangRecord
{
class SlangDecoder
{
public:
    struct ParameterBlock
    {
        const uint8_t* parameterBuffer = nullptr;
        int64_t parameterBufferSize = 0;

        const uint8_t* outputBuffer = nullptr;
        int64_t outputBufferSize = 0;
    };

    struct OutputObject
    {
        ObjectID recorddObjectId;
    };

    SlangDecoder(){};
    ~SlangDecoder(){};

    void addConsumer(IDecoderConsumer* consumer) { m_consumers.add(consumer); }

    bool processMethodCall(FunctionHeader const& header, ParameterBlock const& parameterBlock);
    bool processFunctionCall(FunctionHeader const& header, ParameterBlock const& parameterBlock);

    bool processIGlobalSessionMethods(
        ApiCallId callId,
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    bool processISessionMethods(
        ApiCallId callId,
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    bool processIModuleMethods(
        ApiCallId callId,
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    bool processIEntryPointMethods(
        ApiCallId callId,
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    bool processICompositeComponentTypeMethods(
        ApiCallId callId,
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    bool processITypeConformanceMethods(
        ApiCallId callId,
        ObjectID objectId,
        ParameterBlock const& parameterBlock);

    bool CreateGlobalSession(ParameterBlock const& parameterBlock);
    bool IGlobalSession_createSession(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_findProfile(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_setDownstreamCompilerPath(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_setDownstreamCompilerPrelude(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_getDownstreamCompilerPrelude(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_getBuildTagString(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_setDefaultDownstreamCompiler(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_getDefaultDownstreamCompiler(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_setLanguagePrelude(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_getLanguagePrelude(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_createCompileRequest(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_addBuiltins(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_setSharedLibraryLoader(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_getSharedLibraryLoader(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_checkCompileTargetSupport(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_checkPassThroughSupport(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_compileCoreModule(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_loadCoreModule(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_saveCoreModule(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_findCapability(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IGlobalSession_setDownstreamCompilerForTransition(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_getDownstreamCompilerForTransition(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_getCompilerElapsedTime(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_setSPIRVCoreGrammar(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_parseCommandLineArguments(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IGlobalSession_getSessionDescDigest(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);

    void ISession_getGlobalSession(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_loadModule(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_loadModuleFromBlob(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_loadModuleFromIRBlob(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_loadModuleFromSource(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_loadModuleFromSourceString(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ISession_createCompositeComponentType(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ISession_specializeType(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_getTypeLayout(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_getContainerType(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_getDynamicType(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_getTypeRTTIMangledName(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_getTypeConformanceWitnessMangledName(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ISession_getTypeConformanceWitnessSequentialID(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ISession_createTypeConformanceComponentType(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ISession_createCompileRequest(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_getLoadedModuleCount(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_getLoadedModule(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ISession_isBinaryModuleUpToDate(ObjectID objectId, ParameterBlock const& parameterBlock);

    void IModule_findEntryPointByName(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getDefinedEntryPointCount(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getDefinedEntryPoint(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_serialize(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_writeToFile(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getName(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getFilePath(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getUniqueIdentity(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_findAndCheckEntryPoint(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getSession(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getLayout(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getSpecializationParamCount(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IModule_getEntryPointCode(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getTargetCode(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getResultAsFileSystem(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getEntryPointHash(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_specialize(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_link(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_getEntryPointHostCallable(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_renameEntryPoint(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IModule_linkWithOptions(ObjectID objectId, ParameterBlock const& parameterBlock);

    void IEntryPoint_getSession(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_getLayout(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_getSpecializationParamCount(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IEntryPoint_getEntryPointCode(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_getTargetCode(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_getResultAsFileSystem(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_getEntryPointHash(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_specialize(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_link(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_getEntryPointHostCallable(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void IEntryPoint_renameEntryPoint(ObjectID objectId, ParameterBlock const& parameterBlock);
    void IEntryPoint_linkWithOptions(ObjectID objectId, ParameterBlock const& parameterBlock);

    void ICompositeComponentType_getSession(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_getLayout(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ICompositeComponentType_getSpecializationParamCount(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_getEntryPointCode(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_getTargetCode(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_getResultAsFileSystem(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_getEntryPointHash(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_specialize(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_link(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ICompositeComponentType_getEntryPointHostCallable(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_renameEntryPoint(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ICompositeComponentType_linkWithOptions(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);

    void ITypeConformance_getSession(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ITypeConformance_getLayout(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ITypeConformance_getSpecializationParamCount(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ITypeConformance_getEntryPointCode(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ITypeConformance_getTargetCode(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ITypeConformance_getResultAsFileSystem(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ITypeConformance_getEntryPointHash(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ITypeConformance_specialize(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ITypeConformance_link(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ITypeConformance_getEntryPointHostCallable(
        ObjectID objectId,
        ParameterBlock const& parameterBlock);
    void ITypeConformance_renameEntryPoint(ObjectID objectId, ParameterBlock const& parameterBlock);
    void ITypeConformance_linkWithOptions(ObjectID objectId, ParameterBlock const& parameterBlock);

private:
    Slang::List<IDecoderConsumer*> m_consumers;
};
} // namespace SlangRecord
#endif // SLANG_DECODER_H
