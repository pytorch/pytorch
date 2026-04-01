#include "slang-decoder.h"

#include "../util/record-utility.h"
#include "decoder-helper.h"
#include "parameter-decoder.h"

namespace SlangRecord
{

bool SlangDecoder::processMethodCall(
    FunctionHeader const& header,
    ParameterBlock const& parameterBlock)
{
    ApiClassId classId = static_cast<ApiClassId>(getClassId(header.callId));
    ObjectID objectId = header.handleId;
    switch (classId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang Class Id: %d\n", classId);
        return false;
    case ApiClassId::Class_IGlobalSession:
        return processIGlobalSessionMethods(header.callId, objectId, parameterBlock);
        break;
    case ApiClassId::Class_ISession:
        return processISessionMethods(header.callId, objectId, parameterBlock);
        break;
    case ApiClassId::Class_IModule:
        return processIModuleMethods(header.callId, objectId, parameterBlock);
        break;
    case ApiClassId::Class_IEntryPoint:
        return processIEntryPointMethods(header.callId, objectId, parameterBlock);
        break;
    case ApiClassId::Class_ICompositeComponentType:
        return processICompositeComponentTypeMethods(header.callId, objectId, parameterBlock);
        break;
    case ApiClassId::Class_ITypeConformance:
        return processITypeConformanceMethods(header.callId, objectId, parameterBlock);
        break;
    }
}

bool SlangDecoder::processIGlobalSessionMethods(
    ApiCallId callId,
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    switch (callId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang API call: %d\n", callId);
        break;
    case ApiCallId::IGlobalSession_createSession:
        IGlobalSession_createSession(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_findProfile:
        IGlobalSession_findProfile(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_setDownstreamCompilerPath:
        IGlobalSession_setDownstreamCompilerPath(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_setDownstreamCompilerPrelude:
        IGlobalSession_setDownstreamCompilerPrelude(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getDownstreamCompilerPrelude:
        IGlobalSession_getDownstreamCompilerPrelude(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getBuildTagString:
        IGlobalSession_getBuildTagString(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_setDefaultDownstreamCompiler:
        IGlobalSession_setDefaultDownstreamCompiler(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getDefaultDownstreamCompiler:
        IGlobalSession_getDefaultDownstreamCompiler(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_setLanguagePrelude:
        IGlobalSession_setLanguagePrelude(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getLanguagePrelude:
        IGlobalSession_getLanguagePrelude(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_createCompileRequest:
        IGlobalSession_createCompileRequest(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_addBuiltins:
        IGlobalSession_addBuiltins(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_setSharedLibraryLoader:
        IGlobalSession_setSharedLibraryLoader(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getSharedLibraryLoader:
        IGlobalSession_getSharedLibraryLoader(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_checkCompileTargetSupport:
        IGlobalSession_checkCompileTargetSupport(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_checkPassThroughSupport:
        IGlobalSession_checkPassThroughSupport(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_compileCoreModule:
        IGlobalSession_compileCoreModule(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_loadCoreModule:
        IGlobalSession_loadCoreModule(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_saveCoreModule:
        IGlobalSession_saveCoreModule(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_findCapability:
        IGlobalSession_findCapability(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_setDownstreamCompilerForTransition:
        IGlobalSession_setDownstreamCompilerForTransition(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getDownstreamCompilerForTransition:
        IGlobalSession_getDownstreamCompilerForTransition(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getCompilerElapsedTime:
        IGlobalSession_getCompilerElapsedTime(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_setSPIRVCoreGrammar:
        IGlobalSession_setSPIRVCoreGrammar(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_parseCommandLineArguments:
        IGlobalSession_parseCommandLineArguments(objectId, parameterBlock);
        break;
    case ApiCallId::IGlobalSession_getSessionDescDigest:
        IGlobalSession_getSessionDescDigest(objectId, parameterBlock);
        break;
    }
    return true;
}


bool SlangDecoder::processISessionMethods(
    ApiCallId callId,
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    switch (callId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang API call: %d\n", callId);
        return false;
    case ApiCallId::ISession_getGlobalSession:
        ISession_getGlobalSession(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_loadModule:
        ISession_loadModule(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_loadModuleFromIRBlob:
        ISession_loadModuleFromIRBlob(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_loadModuleFromSource:
        ISession_loadModuleFromSource(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_loadModuleFromSourceString:
        ISession_loadModuleFromSourceString(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_createCompositeComponentType:
        ISession_createCompositeComponentType(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_specializeType:
        ISession_specializeType(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getTypeLayout:
        ISession_getTypeLayout(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getContainerType:
        ISession_getContainerType(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getDynamicType:
        ISession_getDynamicType(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getTypeRTTIMangledName:
        ISession_getTypeRTTIMangledName(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getTypeConformanceWitnessMangledName:
        ISession_getTypeConformanceWitnessMangledName(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getTypeConformanceWitnessSequentialID:
        ISession_getTypeConformanceWitnessSequentialID(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_createTypeConformanceComponentType:
        ISession_createTypeConformanceComponentType(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_createCompileRequest:
        ISession_createCompileRequest(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getLoadedModuleCount:
        ISession_getLoadedModuleCount(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_getLoadedModule:
        ISession_getLoadedModule(objectId, parameterBlock);
        break;
    case ApiCallId::ISession_isBinaryModuleUpToDate:
        ISession_isBinaryModuleUpToDate(objectId, parameterBlock);
        break;
    }
    return true;
}

bool SlangDecoder::processIModuleMethods(
    ApiCallId callId,
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    switch (callId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang API call: %d\n", callId);
        return false;
    case ApiCallId::IModule_findEntryPointByName:
        IModule_findEntryPointByName(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getDefinedEntryPointCount:
        IModule_getDefinedEntryPointCount(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getDefinedEntryPoint:
        IModule_getDefinedEntryPoint(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_serialize:
        IModule_serialize(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_writeToFile:
        IModule_writeToFile(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getName:
        IModule_getName(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getFilePath:
        IModule_getFilePath(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getUniqueIdentity:
        IModule_getUniqueIdentity(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_findAndCheckEntryPoint:
        IModule_findAndCheckEntryPoint(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getSession:
        IModule_getSession(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getLayout:
        IModule_getLayout(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getSpecializationParamCount:
        IModule_getSpecializationParamCount(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getEntryPointCode:
        IModule_getEntryPointCode(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getTargetCode:
        IModule_getTargetCode(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getResultAsFileSystem:
        IModule_getResultAsFileSystem(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getEntryPointHash:
        IModule_getEntryPointHash(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_specialize:
        IModule_specialize(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_link:
        IModule_link(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_getEntryPointHostCallable:
        IModule_getEntryPointHostCallable(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_renameEntryPoint:
        IModule_renameEntryPoint(objectId, parameterBlock);
        break;
    case ApiCallId::IModule_linkWithOptions:
        IModule_linkWithOptions(objectId, parameterBlock);
        break;
    }
    return true;
}

bool SlangDecoder::processIEntryPointMethods(
    ApiCallId callId,
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    switch (callId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang API call: %d\n", callId);
        return false;
    case ApiCallId::IEntryPoint_getSession:
        IEntryPoint_getSession(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_getLayout:
        IEntryPoint_getLayout(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_getSpecializationParamCount:
        IEntryPoint_getSpecializationParamCount(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_getEntryPointCode:
        IEntryPoint_getEntryPointCode(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_getTargetCode:
        IEntryPoint_getTargetCode(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_getResultAsFileSystem:
        IEntryPoint_getResultAsFileSystem(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_getEntryPointHash:
        IEntryPoint_getEntryPointHash(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_specialize:
        IEntryPoint_specialize(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_link:
        IEntryPoint_link(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_getEntryPointHostCallable:
        IEntryPoint_getEntryPointHostCallable(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_renameEntryPoint:
        IEntryPoint_renameEntryPoint(objectId, parameterBlock);
        break;
    case ApiCallId::IEntryPoint_linkWithOptions:
        IEntryPoint_linkWithOptions(objectId, parameterBlock);
        break;
    }
    return true;
}

bool SlangDecoder::processICompositeComponentTypeMethods(
    ApiCallId callId,
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    switch (callId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang API call: %d\n", callId);
        break;
    case ApiCallId::ICompositeComponentType_getSession:
        ICompositeComponentType_getSession(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_getLayout:
        ICompositeComponentType_getLayout(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_getSpecializationParamCount:
        ICompositeComponentType_getSpecializationParamCount(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_getEntryPointCode:
        ICompositeComponentType_getEntryPointCode(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_getTargetCode:
        ICompositeComponentType_getTargetCode(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_getResultAsFileSystem:
        ICompositeComponentType_getResultAsFileSystem(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_getEntryPointHash:
        ICompositeComponentType_getEntryPointHash(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_specialize:
        ICompositeComponentType_specialize(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_link:
        ICompositeComponentType_link(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_getEntryPointHostCallable:
        ICompositeComponentType_getEntryPointHostCallable(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_renameEntryPoint:
        ICompositeComponentType_renameEntryPoint(objectId, parameterBlock);
        break;
    case ApiCallId::ICompositeComponentType_linkWithOptions:
        ICompositeComponentType_linkWithOptions(objectId, parameterBlock);
        break;
    }
    return true;
}

bool SlangDecoder::processITypeConformanceMethods(
    ApiCallId callId,
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    switch (callId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang API call: %d\n", callId);
        return false;
    case ApiCallId::ITypeConformance_getSession:
        ITypeConformance_getSession(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_getLayout:
        ITypeConformance_getLayout(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_getSpecializationParamCount:
        ITypeConformance_getSpecializationParamCount(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_getEntryPointCode:
        ITypeConformance_getEntryPointCode(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_getTargetCode:
        ITypeConformance_getTargetCode(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_getResultAsFileSystem:
        ITypeConformance_getResultAsFileSystem(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_getEntryPointHash:
        ITypeConformance_getEntryPointHash(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_specialize:
        ITypeConformance_specialize(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_link:
        ITypeConformance_link(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_getEntryPointHostCallable:
        ITypeConformance_getEntryPointHostCallable(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_renameEntryPoint:
        ITypeConformance_renameEntryPoint(objectId, parameterBlock);
        break;
    case ApiCallId::ITypeConformance_linkWithOptions:
        ITypeConformance_linkWithOptions(objectId, parameterBlock);
        break;
    }
    return true;
}

bool SlangDecoder::processFunctionCall(
    FunctionHeader const& header,
    ParameterBlock const& parameterBlock)
{
    switch (header.callId)
    {
    default:
        slangRecordLog(LogLevel::Error, "Unhandled Slang API call: %d\n", header.callId);
        return false;
    case ApiCallId::CreateGlobalSession:
        CreateGlobalSession(parameterBlock);
        break;
    }
    return true;
}


bool SlangDecoder::CreateGlobalSession(ParameterBlock const& parameterBlock)
{
    StructDecoder<SlangGlobalSessionDesc> sessionDesc;
    sessionDesc.decode(parameterBlock.parameterBuffer, parameterBlock.parameterBufferSize);

    ObjectID outGlobalSessionId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outGlobalSessionId);

    for (auto consumer : m_consumers)
    {
        consumer->CreateGlobalSession(sessionDesc.getValue(), outGlobalSessionId);
    }
    return true;
}

bool SlangDecoder::IGlobalSession_createSession(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StructDecoder<slang::SessionDesc> sessionDesc;
    sessionDesc.decode(parameterBlock.parameterBuffer, parameterBlock.parameterBufferSize);

    ObjectID outSessionId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSessionId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_createSession(objectId, sessionDesc.getValue(), outSessionId);
    }

    return true;
}

void SlangDecoder::IGlobalSession_findProfile(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StringDecoder name;
    ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        name);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_findProfile(objectId, name.getPointer());
    }
}

void SlangDecoder::IGlobalSession_setDownstreamCompilerPath(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    SlangPassThrough passThrough{};
    readByte = ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        passThrough);
    StringDecoder path;
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        path);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_setDownstreamCompilerPath(
            objectId,
            passThrough,
            path.getPointer());
    }
}

void SlangDecoder::IGlobalSession_setDownstreamCompilerPrelude(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    SlangPassThrough passThrough{};
    readByte = ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        passThrough);
    StringDecoder prelude;
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        prelude);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_setDownstreamCompilerPrelude(
            objectId,
            passThrough,
            prelude.getPointer());
    }
}

void SlangDecoder::IGlobalSession_getDownstreamCompilerPrelude(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    SlangPassThrough passThrough{};
    ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        passThrough);

    ObjectID outPreludeId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outPreludeId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_getDownstreamCompilerPrelude(objectId, passThrough, outPreludeId);
    }
}

void SlangDecoder::IGlobalSession_getBuildTagString(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_setDefaultDownstreamCompiler(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    SlangSourceLanguage sourceLanguage{};
    SlangPassThrough defaultCompiler{};
    readByte = ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        sourceLanguage);
    readByte += ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        defaultCompiler);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_setDefaultDownstreamCompiler(
            objectId,
            sourceLanguage,
            defaultCompiler);
    }
}

void SlangDecoder::IGlobalSession_getDefaultDownstreamCompiler(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    SlangSourceLanguage sourceLanguage{};
    ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        sourceLanguage);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_getDefaultDownstreamCompiler(objectId, sourceLanguage);
    }
}

void SlangDecoder::IGlobalSession_setLanguagePrelude(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    SlangSourceLanguage sourceLanguage{};
    StringDecoder prelude;
    readByte = ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        sourceLanguage);
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        prelude);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_setLanguagePrelude(objectId, sourceLanguage, prelude.getPointer());
    }
}

void SlangDecoder::IGlobalSession_getLanguagePrelude(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    SlangSourceLanguage sourceLanguage{};
    ObjectID outPreludeId = 0;
    ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        sourceLanguage);
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outPreludeId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_getLanguagePrelude(objectId, sourceLanguage, outPreludeId);
    }
}

void SlangDecoder::IGlobalSession_createCompileRequest(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    ObjectID outCompileRequestId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCompileRequestId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_createCompileRequest(objectId, outCompileRequestId);
    }
}

void SlangDecoder::IGlobalSession_addBuiltins(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readBytes = 0;
    StringDecoder sourcePath;
    StringDecoder sourceString;
    readBytes = ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        sourcePath);
    readBytes += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readBytes,
        parameterBlock.parameterBufferSize - readBytes,
        sourceString);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_addBuiltins(
            objectId,
            sourcePath.getPointer(),
            sourceString.getPointer());
    }
}

void SlangDecoder::IGlobalSession_setSharedLibraryLoader(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    // TODO: Not sure if we need to record this function. Because this functions is something like
    // the file system override, it's provided by user code. So capturing it makes no sense. The
    // only way is to wrapper this interface by our own implementation, and record it there.
    slangRecordLog(LogLevel::Error, "%s should not be called\n", __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_getSharedLibraryLoader(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    ObjectID outLoaderId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLoaderId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_getSharedLibraryLoader(objectId, outLoaderId);
    }
}

void SlangDecoder::IGlobalSession_checkCompileTargetSupport(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_checkPassThroughSupport(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_compileCoreModule(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slang::CompileCoreModuleFlags flags{};
    ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        flags);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_compileCoreModule(objectId, flags);
    }
}

void SlangDecoder::IGlobalSession_loadCoreModule(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    PointerDecoder<void*> coreModule;
    ParameterDecoder::decodePointer(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        coreModule);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_loadCoreModule(
            objectId,
            coreModule.getPointer(),
            coreModule.getDataSize());
    }
}

void SlangDecoder::IGlobalSession_saveCoreModule(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    SlangArchiveType archiveType{};
    ObjectID outBlobId = 0;
    ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        archiveType);
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_saveCoreModule(objectId, archiveType, outBlobId);
    }
}

void SlangDecoder::IGlobalSession_findCapability(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_setDownstreamCompilerForTransition(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    SlangCompileTarget source{};
    SlangCompileTarget target{};
    SlangPassThrough compiler{};

    readByte = ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        source);
    readByte += ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        target);
    readByte += ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        compiler);

    for (auto consumer : m_consumers)
    {
        consumer
            ->IGlobalSession_setDownstreamCompilerForTransition(objectId, source, target, compiler);
    }
}

void SlangDecoder::IGlobalSession_getDownstreamCompilerForTransition(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_getCompilerElapsedTime(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_setSPIRVCoreGrammar(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IGlobalSession_parseCommandLineArguments(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    int argc = 0;
    size_t readByte = 0;
    readByte = ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        argc);
    std::vector<char*> argv;

    if (argc > 0)
    {
        uint32_t arrayCount = 0;
        readByte += ParameterDecoder::decodeUint32(
            parameterBlock.parameterBuffer + readByte,
            parameterBlock.parameterBufferSize - readByte,
            arrayCount);

        SLANG_RECORD_ASSERT(arrayCount == (uint32_t)argc);
        argv.resize(arrayCount);

        readByte += ParameterDecoder::decodeStringArray(
            parameterBlock.parameterBuffer + readByte,
            parameterBlock.parameterBufferSize - readByte,
            argv.data(),
            arrayCount);
    }

    ObjectID outSessionDescId = 0;
    ObjectID outAllocationId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSessionDescId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outAllocationId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_parseCommandLineArguments(
            objectId,
            argc,
            argv.data(),
            outSessionDescId,
            outAllocationId);
    }
}

void SlangDecoder::IGlobalSession_getSessionDescDigest(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StructDecoder<slang::SessionDesc> sessionDesc;
    ObjectID outBlobId = 0;
    size_t readByte = 0;
    sessionDesc.decode(parameterBlock.parameterBuffer, parameterBlock.parameterBufferSize);
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->IGlobalSession_getSessionDescDigest(objectId, &sessionDesc.getValue(), outBlobId);
    }
}


void SlangDecoder::ISession_getGlobalSession(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::ISession_loadModule(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    StringDecoder moduleName;
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        moduleName);

    ObjectID outDiagnosticsId = 0;
    ObjectID outModuleId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outModuleId);

    for (auto consumer : m_consumers)
    {
        consumer
            ->ISession_loadModule(objectId, moduleName.getPointer(), outDiagnosticsId, outModuleId);
    }
}

void SlangDecoder::ISession_loadModuleFromIRBlob(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    StringDecoder moduleName;
    StringDecoder path;
    BlobDecoder source;
    readByte = ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        moduleName);
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        path);
    readByte += source.decode(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte);

    ObjectID outDiagnosticsId = 0;
    ObjectID outModuleId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outModuleId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_loadModuleFromIRBlob(
            objectId,
            moduleName.getPointer(),
            path.getPointer(),
            source.getBlob(),
            outDiagnosticsId,
            outModuleId);
    }
}

void SlangDecoder::ISession_loadModuleFromSource(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    StringDecoder moduleName;
    StringDecoder path;
    BlobDecoder source;
    readByte = ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        moduleName);
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        path);
    readByte += source.decode(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte);

    ObjectID outDiagnosticsId = 0;
    ObjectID outModuleId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outModuleId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_loadModuleFromSource(
            objectId,
            moduleName.getPointer(),
            path.getPointer(),
            source.getBlob(),
            outDiagnosticsId,
            outModuleId);
    }
}

void SlangDecoder::ISession_loadModuleFromSourceString(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    StringDecoder moduleName;
    StringDecoder path;
    StringDecoder source;
    readByte = ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        moduleName);
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        path);
    readByte += ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        source);

    ObjectID outDiagnosticsId = 0;
    ObjectID outModuleId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outModuleId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_loadModuleFromSourceString(
            objectId,
            moduleName.getPointer(),
            path.getPointer(),
            source.getPointer(),
            outDiagnosticsId,
            outModuleId);
    }
}

void SlangDecoder::ISession_createCompositeComponentType(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    std::vector<ObjectID> componentTypeIdList;
    uint32_t arrayCount = 0;
    readByte = ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        arrayCount);

    componentTypeIdList.resize(arrayCount);
    readByte += ParameterDecoder::decodeAddressArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        componentTypeIdList.data(),
        arrayCount);

    ObjectID outCompositeComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCompositeComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_createCompositeComponentType(
            objectId,
            componentTypeIdList.data(),
            componentTypeIdList.size(),
            outCompositeComponentTypeId,
            outDiagnosticsId);
    }
}

// TODO: See https://github.com/shader-slang/slang/issues/4624 for more details
void SlangDecoder::ISession_specializeType(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    ObjectID typeId = 0;
    uint32_t arrayCount = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        typeId);
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arrayCount);

    std::vector<slang::SpecializationArg> specializationArgs;
    specializationArgs.resize(arrayCount);
    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        specializationArgs.data(),
        arrayCount);

    ObjectID outDiagnosticsId = 0;
    ObjectID outTypeReflectionId = 0;

    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outTypeReflectionId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_specializeType(
            objectId,
            typeId,
            specializationArgs.data(),
            specializationArgs.size(),
            outDiagnosticsId,
            outTypeReflectionId);
    }
}

void SlangDecoder::ISession_getTypeLayout(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    ObjectID typeId = 0;
    int64_t targetIndex = 0;
    slang::LayoutRules rules{};
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        typeId);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);
    readByte += ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        rules);

    ObjectID outDiagnosticsId = 0;
    ObjectID outTypeLayoutReflectionId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outTypeLayoutReflectionId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_getTypeLayout(
            objectId,
            typeId,
            targetIndex,
            rules,
            outDiagnosticsId,
            outTypeLayoutReflectionId);
    }
}

void SlangDecoder::ISession_getContainerType(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    ObjectID elementType = 0;
    slang::ContainerType containerType{};
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        elementType);
    readByte += ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        containerType);

    ObjectID outDiagnosticsId = 0;
    ObjectID outTypeReflectionId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        outTypeReflectionId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_getContainerType(
            objectId,
            elementType,
            containerType,
            outDiagnosticsId,
            outTypeReflectionId);
    }
}

void SlangDecoder::ISession_getDynamicType(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    ObjectID outTypeReflectionId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outTypeReflectionId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_getDynamicType(objectId, outTypeReflectionId);
    }
}

void SlangDecoder::ISession_getTypeRTTIMangledName(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    ObjectID typeId = 0;
    ObjectID outNameBlobId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        typeId);
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outNameBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_getTypeRTTIMangledName(objectId, typeId, outNameBlobId);
    }
}

void SlangDecoder::ISession_getTypeConformanceWitnessMangledName(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    ObjectID typeId = 0;
    ObjectID interfaceTypeId = 0;
    ObjectID outNameBlobId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        typeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        interfaceTypeId);

    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outNameBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_getTypeConformanceWitnessMangledName(
            objectId,
            typeId,
            interfaceTypeId,
            outNameBlobId);
    }
}

void SlangDecoder::ISession_getTypeConformanceWitnessSequentialID(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;

    ObjectID typeId = 0;
    ObjectID interfaceTypeId = 0;

    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        typeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        interfaceTypeId);

    uint32_t outSequentialId = 0;
    for (auto consumer : m_consumers)
    {
        consumer->ISession_getTypeConformanceWitnessSequentialID(
            objectId,
            typeId,
            interfaceTypeId,
            outSequentialId);
    }
}

void SlangDecoder::ISession_createTypeConformanceComponentType(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection app is not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    ObjectID typeId = 0;
    ObjectID interfaceTypeId = 0;
    int64_t conformanceIdOverride = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        typeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        interfaceTypeId);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        conformanceIdOverride);

    ObjectID outDiagnosticsId = 0;
    ObjectID outConformanceId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize,
        outConformanceId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_createTypeConformanceComponentType(
            objectId,
            typeId,
            interfaceTypeId,
            conformanceIdOverride,
            outConformanceId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ISession_createCompileRequest(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    ObjectID outCompileRequestId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCompileRequestId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_createCompileRequest(objectId, outCompileRequestId);
    }
}

void SlangDecoder::ISession_getLoadedModuleCount(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::ISession_getLoadedModule(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    int64_t index = 0;
    ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        index);

    ObjectID outModuleId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outModuleId);

    for (auto consumer : m_consumers)
    {
        consumer->ISession_getLoadedModule(objectId, index, outModuleId);
    }
}

void SlangDecoder::ISession_isBinaryModuleUpToDate(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}


void SlangDecoder::IModule_findEntryPointByName(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StringDecoder name;
    ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        name);

    ObjectID outEntryPointId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outEntryPointId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_findEntryPointByName(objectId, name.getPointer(), outEntryPointId);
    }
}

void SlangDecoder::IModule_getDefinedEntryPointCount(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IModule_getDefinedEntryPoint(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    int32_t index;
    ObjectID outEntryPointId = 0;
    ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        index);
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outEntryPointId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_getDefinedEntryPoint(objectId, index, outEntryPointId);
    }
}

void SlangDecoder::IModule_serialize(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    ObjectID outSerializedBlobId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSerializedBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_serialize(objectId, outSerializedBlobId);
    }
}

void SlangDecoder::IModule_writeToFile(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    StringDecoder fileName;
    ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        fileName);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_writeToFile(objectId, fileName.getPointer());
    }
}

void SlangDecoder::IModule_getName(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IModule_getFilePath(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IModule_getUniqueIdentity(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IModule_findAndCheckEntryPoint(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StringDecoder name;
    SlangStage stage{};
    size_t readByte = 0;
    readByte = ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        name);
    readByte += ParameterDecoder::decodeEnumValue(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        stage);

    ObjectID outEntryPointId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outEntryPointId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_findAndCheckEntryPoint(
            objectId,
            name.getPointer(),
            stage,
            outEntryPointId,
            outDiagnosticsId);
    }
}

void SlangDecoder::IModule_getSession(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IModule_getLayout(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    int64_t targetIndex = 0;
    ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);
    ObjectID outDiagnosticsId = 0;
    ObjectID programLayoutId = 0;

    size_t readByte = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        programLayoutId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_getLayout(objectId, targetIndex, outDiagnosticsId, programLayoutId);
    }
}

void SlangDecoder::IModule_getSpecializationParamCount(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IModule_getEntryPointCode(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_getEntryPointCode(
            objectId,
            entryPointIndex,
            targetIndex,
            outCodeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::IModule_getTargetCode(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
    }
}

void SlangDecoder::IModule_getResultAsFileSystem(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outFileSystemId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outFileSystemId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_getResultAsFileSystem(
            objectId,
            entryPointIndex,
            targetIndex,
            outFileSystemId);
    }
}

void SlangDecoder::IModule_getEntryPointHash(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outBlobId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_getEntryPointHash(objectId, entryPointIndex, targetIndex, outBlobId);
    }
}

void SlangDecoder::IModule_specialize(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection interfaces are not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    int64_t specializationArgCount = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        specializationArgCount);

    std::vector<slang::SpecializationArg> specializationArgs;

    uint32_t arraySize = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arraySize);

    SLANG_RECORD_ASSERT(arraySize == specializationArgCount);

    specializationArgs.resize(specializationArgCount);
    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        specializationArgs.data(),
        specializationArgCount);

    ObjectID outSpecializedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSpecializedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_specialize(
            objectId,
            specializationArgs.data(),
            specializationArgCount,
            outSpecializedComponentTypeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::IModule_link(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
    }
}

void SlangDecoder::IModule_getEntryPointHostCallable(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int32_t entryPointIndex = 0;
    int32_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outSharedLibraryId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSharedLibraryId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_getEntryPointHostCallable(
            objectId,
            entryPointIndex,
            targetIndex,
            outSharedLibraryId,
            outDiagnosticsId);
    }
}

void SlangDecoder::IModule_renameEntryPoint(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    StringDecoder newName;
    ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        newName);

    ObjectID outEntryPointId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outEntryPointId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_renameEntryPoint(objectId, newName.getPointer(), outEntryPointId);
    }
}

void SlangDecoder::IModule_linkWithOptions(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    uint32_t compilerOptionEntryCount = 0;
    readByte = ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        compilerOptionEntryCount);

    std::vector<slang::CompilerOptionEntry> compilerOptionEntries;

    uint32_t arrayCount = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arrayCount);

    SLANG_RECORD_ASSERT(arrayCount == compilerOptionEntryCount);
    compilerOptionEntries.resize(compilerOptionEntryCount);

    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        compilerOptionEntries.data(),
        compilerOptionEntryCount);

    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_linkWithOptions(
            objectId,
            outLinkedComponentTypeId,
            compilerOptionEntryCount,
            compilerOptionEntries.data(),
            outDiagnosticsId);
    }
}

void SlangDecoder::IEntryPoint_getSession(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IEntryPoint_getLayout(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    int64_t targetIndex = 0;
    ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);

    ObjectID outDiagnosticsId = 0;
    ObjectID programLayoutId = 0;

    size_t readByte = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        programLayoutId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_getLayout(objectId, targetIndex, outDiagnosticsId, programLayoutId);
    }
}

void SlangDecoder::IEntryPoint_getSpecializationParamCount(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::IEntryPoint_getEntryPointCode(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_getEntryPointCode(
            objectId,
            entryPointIndex,
            targetIndex,
            outCodeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::IEntryPoint_getTargetCode(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
    }
}

void SlangDecoder::IEntryPoint_getResultAsFileSystem(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outFileSystemId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outFileSystemId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_getResultAsFileSystem(
            objectId,
            entryPointIndex,
            targetIndex,
            outFileSystemId);
    }
}

void SlangDecoder::IEntryPoint_getEntryPointHash(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outBlobId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_getEntryPointHash(objectId, entryPointIndex, targetIndex, outBlobId);
    }
}

void SlangDecoder::IEntryPoint_specialize(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection interfaces are not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    int64_t specializationArgCount = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        specializationArgCount);

    std::vector<slang::SpecializationArg> specializationArgs;

    uint32_t arraySize = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arraySize);

    SLANG_RECORD_ASSERT(arraySize == specializationArgCount);

    specializationArgs.resize(specializationArgCount);
    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        specializationArgs.data(),
        specializationArgCount);

    ObjectID outSpecializedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSpecializedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_specialize(
            objectId,
            specializationArgs.data(),
            specializationArgCount,
            outSpecializedComponentTypeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::IEntryPoint_link(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
    }
}

void SlangDecoder::IEntryPoint_getEntryPointHostCallable(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int32_t entryPointIndex = 0;
    int32_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outSharedLibraryId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSharedLibraryId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_getEntryPointHostCallable(
            objectId,
            entryPointIndex,
            targetIndex,
            outSharedLibraryId,
            outDiagnosticsId);
    }
}

void SlangDecoder::IEntryPoint_renameEntryPoint(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StringDecoder newName;
    ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        newName);

    ObjectID outEntryPointId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outEntryPointId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_renameEntryPoint(objectId, newName.getPointer(), outEntryPointId);
    }
}

void SlangDecoder::IEntryPoint_linkWithOptions(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    uint32_t compilerOptionEntryCount = 0;
    readByte = ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        compilerOptionEntryCount);

    std::vector<slang::CompilerOptionEntry> compilerOptionEntries;

    uint32_t arrayCount = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arrayCount);

    SLANG_RECORD_ASSERT(arrayCount == compilerOptionEntryCount);
    compilerOptionEntries.resize(compilerOptionEntryCount);

    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        compilerOptionEntries.data(),
        compilerOptionEntryCount);

    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IEntryPoint_linkWithOptions(
            objectId,
            outLinkedComponentTypeId,
            compilerOptionEntryCount,
            compilerOptionEntries.data(),
            outDiagnosticsId);
    }
}


void SlangDecoder::ICompositeComponentType_getSession(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::ICompositeComponentType_getLayout(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    int64_t targetIndex = 0;
    ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);
    ObjectID outDiagnosticsId = 0;
    ObjectID programLayoutId = 0;

    size_t readByte = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        programLayoutId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_getLayout(
            objectId,
            targetIndex,
            outDiagnosticsId,
            programLayoutId);
    }
}

void SlangDecoder::ICompositeComponentType_getSpecializationParamCount(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::ICompositeComponentType_getEntryPointCode(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_getEntryPointCode(
            objectId,
            entryPointIndex,
            targetIndex,
            outCodeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ICompositeComponentType_getTargetCode(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_getTargetCode(
            objectId,
            targetIndex,
            outCodeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ICompositeComponentType_getResultAsFileSystem(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outFileSystemId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outFileSystemId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_getResultAsFileSystem(
            objectId,
            entryPointIndex,
            targetIndex,
            outFileSystemId);
    }
}

void SlangDecoder::ICompositeComponentType_getEntryPointHash(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outBlobId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outBlobId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_getEntryPointHash(
            objectId,
            entryPointIndex,
            targetIndex,
            outBlobId);
    }
}

void SlangDecoder::ICompositeComponentType_specialize(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection interfaces are not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    int64_t specializationArgCount = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        specializationArgCount);

    std::vector<slang::SpecializationArg> specializationArgs;

    uint32_t arraySize = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arraySize);

    SLANG_RECORD_ASSERT(arraySize == specializationArgCount);

    specializationArgs.resize(specializationArgCount);
    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        specializationArgs.data(),
        specializationArgCount);

    ObjectID outSpecializedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSpecializedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_specialize(
            objectId,
            specializationArgs.data(),
            specializationArgCount,
            outSpecializedComponentTypeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ICompositeComponentType_link(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_link(
            objectId,
            outLinkedComponentTypeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ICompositeComponentType_getEntryPointHostCallable(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int32_t entryPointIndex = 0;
    int32_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outSharedLibraryId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSharedLibraryId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_getEntryPointHostCallable(
            objectId,
            entryPointIndex,
            targetIndex,
            outSharedLibraryId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ICompositeComponentType_renameEntryPoint(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StringDecoder newName;
    ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        newName);

    ObjectID outEntryPointId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outEntryPointId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_renameEntryPoint(
            objectId,
            newName.getPointer(),
            outEntryPointId);
    }
}

void SlangDecoder::ICompositeComponentType_linkWithOptions(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    uint32_t compilerOptionEntryCount = 0;
    readByte = ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        compilerOptionEntryCount);

    std::vector<slang::CompilerOptionEntry> compilerOptionEntries;

    uint32_t arrayCount = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arrayCount);

    SLANG_RECORD_ASSERT(arrayCount == compilerOptionEntryCount);
    compilerOptionEntries.resize(compilerOptionEntryCount);

    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        compilerOptionEntries.data(),
        compilerOptionEntryCount);

    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ICompositeComponentType_linkWithOptions(
            objectId,
            outLinkedComponentTypeId,
            compilerOptionEntryCount,
            compilerOptionEntries.data(),
            outDiagnosticsId);
    }
}


void SlangDecoder::ITypeConformance_getSession(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::ITypeConformance_getLayout(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    int64_t targetIndex = 0;
    ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);
    ObjectID outDiagnosticsId = 0;
    ObjectID programLayoutId = 0;

    size_t readByte = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outDiagnosticsId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        programLayoutId);

    for (auto consumer : m_consumers)
    {
        consumer
            ->ITypeConformance_getLayout(objectId, targetIndex, outDiagnosticsId, programLayoutId);
    }
}

void SlangDecoder::ITypeConformance_getSpecializationParamCount(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    (void)objectId;
    (void)parameterBlock;
    slangRecordLog(
        LogLevel::Debug,
        "%s should not be called, it'a not recordd\n",
        __PRETTY_FUNCTION__);
}

void SlangDecoder::ITypeConformance_getEntryPointCode(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ITypeConformance_getEntryPointCode(
            objectId,
            entryPointIndex,
            targetIndex,
            outCodeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ITypeConformance_getTargetCode(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        targetIndex);

    ObjectID outCodeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outCodeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer
            ->ITypeConformance_getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
    }
}

void SlangDecoder::ITypeConformance_getResultAsFileSystem(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outFileSystemId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outFileSystemId);

    for (auto consumer : m_consumers)
    {
        consumer->ITypeConformance_getResultAsFileSystem(
            objectId,
            entryPointIndex,
            targetIndex,
            outFileSystemId);
    }
}

void SlangDecoder::ITypeConformance_getEntryPointHash(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int64_t entryPointIndex = 0;
    int64_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outBlobId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outBlobId);

    for (auto consumer : m_consumers)
    {
        consumer
            ->ITypeConformance_getEntryPointHash(objectId, entryPointIndex, targetIndex, outBlobId);
    }
}

void SlangDecoder::ITypeConformance_specialize(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    slangRecordLog(
        LogLevel::Error,
        "%s: The shader reflection interfaces are not recordd\n",
        __PRETTY_FUNCTION__);

    size_t readByte = 0;
    int64_t specializationArgCount = 0;
    readByte = ParameterDecoder::decodeInt64(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        specializationArgCount);

    std::vector<slang::SpecializationArg> specializationArgs;

    uint32_t arraySize = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arraySize);

    SLANG_RECORD_ASSERT(arraySize == specializationArgCount);

    specializationArgs.resize(specializationArgCount);
    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        specializationArgs.data(),
        specializationArgCount);

    ObjectID outSpecializedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSpecializedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->IModule_specialize(
            objectId,
            specializationArgs.data(),
            specializationArgCount,
            outSpecializedComponentTypeId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ITypeConformance_link(ObjectID objectId, ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ITypeConformance_link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
    }
}

void SlangDecoder::ITypeConformance_getEntryPointHostCallable(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    int32_t entryPointIndex = 0;
    int32_t targetIndex = 0;
    readByte = ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        entryPointIndex);
    readByte += ParameterDecoder::decodeInt32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        targetIndex);

    ObjectID outSharedLibraryId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outSharedLibraryId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ITypeConformance_getEntryPointHostCallable(
            objectId,
            entryPointIndex,
            targetIndex,
            outSharedLibraryId,
            outDiagnosticsId);
    }
}

void SlangDecoder::ITypeConformance_renameEntryPoint(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    StringDecoder newName;
    ParameterDecoder::decodeString(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        newName);

    ObjectID outEntryPointId = 0;
    ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outEntryPointId);

    for (auto consumer : m_consumers)
    {
        consumer->ITypeConformance_renameEntryPoint(
            objectId,
            newName.getPointer(),
            outEntryPointId);
    }
}

void SlangDecoder::ITypeConformance_linkWithOptions(
    ObjectID objectId,
    ParameterBlock const& parameterBlock)
{
    size_t readByte = 0;
    uint32_t compilerOptionEntryCount = 0;
    readByte = ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer,
        parameterBlock.parameterBufferSize,
        compilerOptionEntryCount);

    std::vector<slang::CompilerOptionEntry> compilerOptionEntries;

    uint32_t arrayCount = 0;
    readByte += ParameterDecoder::decodeUint32(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        arrayCount);

    SLANG_RECORD_ASSERT(arrayCount == compilerOptionEntryCount);
    compilerOptionEntries.resize(compilerOptionEntryCount);

    readByte += ParameterDecoder::decodeStructArray(
        parameterBlock.parameterBuffer + readByte,
        parameterBlock.parameterBufferSize - readByte,
        compilerOptionEntries.data(),
        compilerOptionEntryCount);

    ObjectID outLinkedComponentTypeId = 0;
    ObjectID outDiagnosticsId = 0;
    readByte = ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer,
        parameterBlock.outputBufferSize,
        outLinkedComponentTypeId);
    readByte += ParameterDecoder::decodeAddress(
        parameterBlock.outputBuffer + readByte,
        parameterBlock.outputBufferSize - readByte,
        outDiagnosticsId);

    for (auto consumer : m_consumers)
    {
        consumer->ITypeConformance_linkWithOptions(
            objectId,
            outLinkedComponentTypeId,
            compilerOptionEntryCount,
            compilerOptionEntries.data(),
            outDiagnosticsId);
    }
}
} // namespace SlangRecord
