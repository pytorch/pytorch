#ifndef SLANG_TYPE_SYSTEM_SHARED_H
#define SLANG_TYPE_SYSTEM_SHARED_H

#include "slang.h"

namespace Slang
{
#define FOREACH_BASE_TYPE(X) \
    X(Void)                  \
    X(Bool)                  \
    X(Int8)                  \
    X(Int16)                 \
    X(Int)                   \
    X(Int64)                 \
    X(UInt8)                 \
    X(UInt16)                \
    X(UInt)                  \
    X(UInt64)                \
    X(Half)                  \
    X(Float)                 \
    X(Double)                \
    X(Char)                  \
    X(IntPtr)                \
    X(UIntPtr)               \
    /* end */

enum class BaseType
{
#define DEFINE_BASE_TYPE(NAME) NAME,
    FOREACH_BASE_TYPE(DEFINE_BASE_TYPE)
#undef DEFINE_BASE_TYPE

        CountOf,
};

enum class SamplerStateFlavor : uint8_t
{
    SamplerState,
    SamplerComparisonState,
};

const int kCoreModule_ResourceAccessReadOnly = 0;
const int kCoreModule_ResourceAccessReadWrite = 1;
const int kCoreModule_ResourceAccessWriteOnly = 2;
const int kCoreModule_ResourceAccessRasterizerOrdered = 3;
const int kCoreModule_ResourceAccessFeedback = 4;

const int kCoreModule_ShapeIndex1D = 0;
const int kCoreModule_ShapeIndex2D = 1;
const int kCoreModule_ShapeIndex3D = 2;
const int kCoreModule_ShapeIndexCube = 3;
const int kCoreModule_ShapeIndexBuffer = 4;

const int kCoreModule_TextureShapeParameterIndex = 1;
const int kCoreModule_TextureIsArrayParameterIndex = 2;
const int kCoreModule_TextureIsMultisampleParameterIndex = 3;
const int kCoreModule_TextureSampleCountParameterIndex = 4;
const int kCoreModule_TextureAccessParameterIndex = 5;
const int kCoreModule_TextureIsShadowParameterIndex = 6;
const int kCoreModule_TextureIsCombinedParameterIndex = 7;
const int kCoreModule_TextureFormatParameterIndex = 8;

enum class AddressSpace : uint64_t
{
    Generic = 0x7fffffff,
    // Corresponds to SPIR-V's SpvStorageClassPrivate
    ThreadLocal = 1,
    Global,
    // Corresponds to SPIR-V's SpvStorageClassWorkgroup
    GroupShared,
    // Corresponds to SPIR-V's SpvStorageClassUniform
    Uniform,
    // specific address space for payload data in metal
    MetalObjectData,
    // Corresponds to SPIR-V's SpvStorageClassInput
    Input,
    // Same as `Input`, but used for builtin input variables.
    BuiltinInput,
    // Corresponds to SPIR-V's SpvStorageClassOutput
    Output,
    // Same as `Output`, but used for builtin output variables.
    BuiltinOutput,
    // Corresponds to SPIR-V's SpvStorageClassTaskPayloadWorkgroupEXT
    TaskPayloadWorkgroup,
    // Corresponds to SPIR-V's SpvStorageClassFunction
    Function,
    // Corresponds to SPIR-V's SpvStorageClassStorageBuffer
    StorageBuffer,
    // Corresponds to SPIR-V's SpvStorageClassPushConstant,
    PushConstant,
    // Corresponds to SPIR-V's SpvStorageClassRayPayloadKHR,
    RayPayloadKHR,
    // Corresponds to SPIR-V's SpvStorageClassIncomingRayPayloadKHR,
    IncomingRayPayload,
    // Corresponds to SPIR-V's SpvStorageClassCallableDataKHR
    CallableDataKHR,
    // Corresponds to SPIR-V's SpvStorageClassIncomingCallableDataKHR
    IncomingCallableData,
    // Corresponds to SPIR-V's SpvStorageClassHitObjectAttributeNV,
    HitObjectAttribute,
    // Corresponds to SPIR-V's SpvStorageClassHitAttributeKHR,
    HitAttribute,
    // Corresponds to SPIR-V's SpvStorageClassShaderRecordBufferKHR,
    ShaderRecordBuffer,
    // Corresponds to SPIR-V's SpvStorageClassUniformConstant,
    UniformConstant,
    // Corresponds to SPIR-V's SpvStorageClassImage
    Image,
    // Represents a SPIR-V specialization constant
    SpecializationConstant,
    // Corresponds to SPIR-V's SpvStorageClassNodePayloadAMDX,
    NodePayloadAMDX,

    // Default address space for a user-defined pointer
    UserPointer = 0x100000001ULL,
};
} // namespace Slang

#endif
