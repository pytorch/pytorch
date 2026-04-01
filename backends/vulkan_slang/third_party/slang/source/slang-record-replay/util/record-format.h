#ifndef API_CALL_ID_H
#define API_CALL_ID_H

#include <cstdint>

namespace SlangRecord
{
constexpr uint32_t makeApiCallId(uint16_t classId, uint16_t memberFunctionId)
{
    return ((static_cast<uint32_t>(classId) << 16) & 0xffff0000) |
           (static_cast<uint32_t>(memberFunctionId) & 0x0000ffff);
}

constexpr uint16_t getClassId(uint32_t callId)
{
    return static_cast<uint16_t>((callId >> 16) & 0x0000ffff);
}

constexpr uint16_t getMemberFunctionId(uint32_t callId)
{
    return static_cast<uint16_t>(callId & 0x0000ffff);
}

enum ApiClassId : uint16_t
{
    GlobalFunction = 1,
    Class_IGlobalSession = 2,
    Class_ISession = 3,
    Class_IModule = 4,
    Class_IEntryPoint = 5,
    Class_ICompositeComponentType = 6,
    Class_ITypeConformance = 7,
    Unknown = 0xFFFF
};

// Store the pointer value in a 64-bit integer
typedef uint64_t AddressFormat;

// Use the address directly to represent the slang object.
typedef AddressFormat ObjectID;

constexpr uint64_t g_globalFunctionHandle = 0;
constexpr uint32_t MAGIC_HEADER = 0x44414548;
constexpr uint32_t MAGIC_TAILER = 0x4C494154;

enum IComponentTypeMethodId : uint16_t
{
    getSession = 0x000A,
    getLayout = 0x000B,
    getSpecializationParamCount = 0x000C,
    getEntryPointCode = 0x000D,
    getTargetCode = 0x000E,
    getResultAsFileSystem = 0x000F,
    getEntryPointHash = 0x0010,
    specialize = 0x0011,
    link = 0x0012,
    getEntryPointHostCallable = 0x0013,
    renameEntryPoint = 0x0014,
    linkWithOptions = 0x0015,
};

enum ApiCallId : uint32_t
{
    InvalidCallId = 0x00000000,
    CreateGlobalSession = makeApiCallId(GlobalFunction, 0x0000),
    IGlobalSession_createSession = makeApiCallId(Class_IGlobalSession, 0x0001),
    IGlobalSession_findProfile = makeApiCallId(Class_IGlobalSession, 0x0002),
    IGlobalSession_setDownstreamCompilerPath = makeApiCallId(Class_IGlobalSession, 0x0003),
    IGlobalSession_setDownstreamCompilerPrelude = makeApiCallId(Class_IGlobalSession, 0x0004),
    IGlobalSession_getDownstreamCompilerPrelude = makeApiCallId(Class_IGlobalSession, 0x0005),
    IGlobalSession_getBuildTagString = makeApiCallId(Class_IGlobalSession, 0x0006),
    IGlobalSession_setDefaultDownstreamCompiler = makeApiCallId(Class_IGlobalSession, 0x0007),
    IGlobalSession_getDefaultDownstreamCompiler = makeApiCallId(Class_IGlobalSession, 0x0008),
    IGlobalSession_setLanguagePrelude = makeApiCallId(Class_IGlobalSession, 0x0009),
    IGlobalSession_getLanguagePrelude = makeApiCallId(Class_IGlobalSession, 0x000A),
    IGlobalSession_createCompileRequest = makeApiCallId(Class_IGlobalSession, 0x000B),
    IGlobalSession_addBuiltins = makeApiCallId(Class_IGlobalSession, 0x000C),
    IGlobalSession_setSharedLibraryLoader = makeApiCallId(Class_IGlobalSession, 0x000D),
    IGlobalSession_getSharedLibraryLoader = makeApiCallId(Class_IGlobalSession, 0x000E),
    IGlobalSession_checkCompileTargetSupport = makeApiCallId(Class_IGlobalSession, 0x000F),
    IGlobalSession_checkPassThroughSupport = makeApiCallId(Class_IGlobalSession, 0x0010),
    IGlobalSession_compileCoreModule = makeApiCallId(Class_IGlobalSession, 0x0011),
    IGlobalSession_loadCoreModule = makeApiCallId(Class_IGlobalSession, 0x0012),
    IGlobalSession_saveCoreModule = makeApiCallId(Class_IGlobalSession, 0x0013),
    IGlobalSession_findCapability = makeApiCallId(Class_IGlobalSession, 0x0014),
    IGlobalSession_setDownstreamCompilerForTransition = makeApiCallId(Class_IGlobalSession, 0x0015),
    IGlobalSession_getDownstreamCompilerForTransition = makeApiCallId(Class_IGlobalSession, 0x0016),
    IGlobalSession_getCompilerElapsedTime = makeApiCallId(Class_IGlobalSession, 0x0017),
    IGlobalSession_setSPIRVCoreGrammar = makeApiCallId(Class_IGlobalSession, 0x0018),
    IGlobalSession_parseCommandLineArguments = makeApiCallId(Class_IGlobalSession, 0x0019),
    IGlobalSession_getSessionDescDigest = makeApiCallId(Class_IGlobalSession, 0x001A),
    IGlobalSession_compileBuiltinModule = makeApiCallId(Class_IGlobalSession, 0x001B),
    IGlobalSession_loadBuiltinModule = makeApiCallId(Class_IGlobalSession, 0x001C),
    IGlobalSession_saveBuiltinModule = makeApiCallId(Class_IGlobalSession, 0x001D),

    ISession_getGlobalSession = makeApiCallId(Class_ISession, 0x0001),
    ISession_loadModule = makeApiCallId(Class_ISession, 0x0002),
    ISession_loadModuleFromIRBlob = makeApiCallId(Class_ISession, 0x0004),
    ISession_loadModuleFromSource = makeApiCallId(Class_ISession, 0x0005),
    ISession_loadModuleFromSourceString = makeApiCallId(Class_ISession, 0x0006),
    ISession_createCompositeComponentType = makeApiCallId(Class_ISession, 0x0007),
    ISession_specializeType = makeApiCallId(Class_ISession, 0x0008),
    ISession_getTypeLayout = makeApiCallId(Class_ISession, 0x0009),
    ISession_getContainerType = makeApiCallId(Class_ISession, 0x000A),
    ISession_getDynamicType = makeApiCallId(Class_ISession, 0x000B),
    ISession_getTypeRTTIMangledName = makeApiCallId(Class_ISession, 0x000C),
    ISession_getTypeConformanceWitnessMangledName = makeApiCallId(Class_ISession, 0x000D),
    ISession_getTypeConformanceWitnessSequentialID = makeApiCallId(Class_ISession, 0x000E),
    ISession_createTypeConformanceComponentType = makeApiCallId(Class_ISession, 0x000F),
    ISession_createCompileRequest = makeApiCallId(Class_ISession, 0x0010),
    ISession_getLoadedModuleCount = makeApiCallId(Class_ISession, 0x0011),
    ISession_getLoadedModule = makeApiCallId(Class_ISession, 0x0012),
    ISession_isBinaryModuleUpToDate = makeApiCallId(Class_ISession, 0x0013),


    IModule_findEntryPointByName = makeApiCallId(Class_IModule, 0x0001),
    IModule_getDefinedEntryPointCount = makeApiCallId(Class_IModule, 0x0002),
    IModule_getDefinedEntryPoint = makeApiCallId(Class_IModule, 0x0003),
    IModule_serialize = makeApiCallId(Class_IModule, 0x0004),
    IModule_writeToFile = makeApiCallId(Class_IModule, 0x0005),
    IModule_getName = makeApiCallId(Class_IModule, 0x0006),
    IModule_getFilePath = makeApiCallId(Class_IModule, 0x0007),
    IModule_getUniqueIdentity = makeApiCallId(Class_IModule, 0x0008),
    IModule_findAndCheckEntryPoint = makeApiCallId(Class_IModule, 0x0009),

    IModule_getSession = makeApiCallId(Class_IModule, IComponentTypeMethodId::getSession),
    IModule_getLayout = makeApiCallId(Class_IModule, IComponentTypeMethodId::getLayout),
    IModule_getSpecializationParamCount =
        makeApiCallId(Class_IModule, IComponentTypeMethodId::getSpecializationParamCount),
    IModule_getEntryPointCode =
        makeApiCallId(Class_IModule, IComponentTypeMethodId::getEntryPointCode),
    IModule_getTargetCode = makeApiCallId(Class_IModule, IComponentTypeMethodId::getTargetCode),
    IModule_getResultAsFileSystem =
        makeApiCallId(Class_IModule, IComponentTypeMethodId::getResultAsFileSystem),
    IModule_getEntryPointHash =
        makeApiCallId(Class_IModule, IComponentTypeMethodId::getEntryPointHash),
    IModule_specialize = makeApiCallId(Class_IModule, IComponentTypeMethodId::specialize),
    IModule_link = makeApiCallId(Class_IModule, IComponentTypeMethodId::link),
    IModule_getEntryPointHostCallable =
        makeApiCallId(Class_IModule, IComponentTypeMethodId::getEntryPointHostCallable),
    IModule_renameEntryPoint =
        makeApiCallId(Class_IModule, IComponentTypeMethodId::renameEntryPoint),
    IModule_linkWithOptions = makeApiCallId(Class_IModule, IComponentTypeMethodId::linkWithOptions),


    IEntryPoint_getSession = makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getSession),
    IEntryPoint_getLayout = makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getLayout),
    IEntryPoint_getSpecializationParamCount =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getSpecializationParamCount),
    IEntryPoint_getEntryPointCode =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getEntryPointCode),
    IEntryPoint_getTargetCode =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getTargetCode),
    IEntryPoint_getResultAsFileSystem =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getResultAsFileSystem),
    IEntryPoint_getEntryPointHash =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getEntryPointHash),
    IEntryPoint_specialize = makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::specialize),
    IEntryPoint_link = makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::link),
    IEntryPoint_getEntryPointHostCallable =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::getEntryPointHostCallable),
    IEntryPoint_renameEntryPoint =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::renameEntryPoint),
    IEntryPoint_linkWithOptions =
        makeApiCallId(Class_IEntryPoint, IComponentTypeMethodId::linkWithOptions),

    ICompositeComponentType_getSession =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::getSession),
    ICompositeComponentType_getLayout =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::getLayout),
    ICompositeComponentType_getSpecializationParamCount = makeApiCallId(
        Class_ICompositeComponentType,
        IComponentTypeMethodId::getSpecializationParamCount),
    ICompositeComponentType_getEntryPointCode =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::getEntryPointCode),
    ICompositeComponentType_getTargetCode =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::getTargetCode),
    ICompositeComponentType_getResultAsFileSystem =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::getResultAsFileSystem),
    ICompositeComponentType_getEntryPointHash =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::getEntryPointHash),
    ICompositeComponentType_specialize =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::specialize),
    ICompositeComponentType_link =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::link),
    ICompositeComponentType_getEntryPointHostCallable = makeApiCallId(
        Class_ICompositeComponentType,
        IComponentTypeMethodId::getEntryPointHostCallable),
    ICompositeComponentType_renameEntryPoint =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::renameEntryPoint),
    ICompositeComponentType_linkWithOptions =
        makeApiCallId(Class_ICompositeComponentType, IComponentTypeMethodId::linkWithOptions),

    ITypeConformance_getSession =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getSession),
    ITypeConformance_getLayout =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getLayout),
    ITypeConformance_getSpecializationParamCount =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getSpecializationParamCount),
    ITypeConformance_getEntryPointCode =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getEntryPointCode),
    ITypeConformance_getTargetCode =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getTargetCode),
    ITypeConformance_getResultAsFileSystem =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getResultAsFileSystem),
    ITypeConformance_getEntryPointHash =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getEntryPointHash),
    ITypeConformance_specialize =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::specialize),
    ITypeConformance_link = makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::link),
    ITypeConformance_getEntryPointHostCallable =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::getEntryPointHostCallable),
    ITypeConformance_renameEntryPoint =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::renameEntryPoint),
    ITypeConformance_linkWithOptions =
        makeApiCallId(Class_ITypeConformance, IComponentTypeMethodId::linkWithOptions),
};

struct FunctionHeader
{
    uint32_t magic{MAGIC_HEADER};
    ApiCallId callId{InvalidCallId};
    ObjectID handleId{0};
    uint64_t dataSizeInBytes{0};
    uint64_t threadId{0};
};

struct FunctionTailer
{
    uint32_t magic{MAGIC_TAILER};
    uint32_t dataSizeInBytes{0};
};

} // namespace SlangRecord
#endif
