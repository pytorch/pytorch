// debug-helper-functions.h
#pragma once
#include "debug-base.h"
#include "debug-buffer.h"
#include "debug-command-buffer.h"
#include "debug-command-queue.h"
#include "debug-device.h"
#include "debug-fence.h"
#include "debug-framebuffer.h"
#include "debug-pipeline-state.h"
#include "debug-query.h"
#include "debug-render-pass.h"
#include "debug-resource-views.h"
#include "debug-sampler-state.h"
#include "debug-shader-object.h"
#include "debug-shader-program.h"
#include "debug-shader-table.h"
#include "debug-swap-chain.h"
#include "debug-texture.h"
#include "debug-transient-heap.h"
#include "debug-vertex-layout.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

#ifdef __FUNCSIG__
#define SLANG_FUNC_SIG __FUNCSIG__
#elif defined(__PRETTY_FUNCTION__)
#define SLANG_FUNC_SIG __FUNCSIG__
#elif defined(__FUNCTION__)
#define SLANG_FUNC_SIG __FUNCTION__
#else
#define SLANG_FUNC_SIG "UnknownFunction"
#endif

extern thread_local const char* _currentFunctionName;
struct SetCurrentFuncRAII
{
    SetCurrentFuncRAII(const char* funcName) { _currentFunctionName = funcName; }
    ~SetCurrentFuncRAII() { _currentFunctionName = nullptr; }
};
#define SLANG_GFX_API_FUNC SetCurrentFuncRAII setFuncNameRAII(SLANG_FUNC_SIG)
#define SLANG_GFX_API_FUNC_NAME(x) SetCurrentFuncRAII setFuncNameRAII(x)

/// Returns the public API function name from a `SLANG_FUNC_SIG` string.
String _gfxGetFuncName(const char* input);

template<typename... TArgs>
char* _gfxDiagnoseFormat(
    char* buffer,            // Initial buffer to output formatted string.
    size_t shortBufferSize,  // Size of the initial buffer.
    List<char>& bufferArray, // A list for allocating a large buffer if needed.
    const char* format,      // The format string.
    TArgs... args)
{
    int length = sprintf_s(buffer, shortBufferSize, format, args...);
    if (length < 0)
        return buffer;
    if (length > 255)
    {
        bufferArray.setCount(length + 1);
        buffer = bufferArray.getBuffer();
        sprintf_s(buffer, bufferArray.getCount(), format, args...);
    }
    return buffer;
}

template<typename... TArgs>
void _gfxDiagnoseImpl(DebugMessageType type, const char* format, TArgs... args)
{
    char shortBuffer[256];
    List<char> bufferArray;
    auto buffer =
        _gfxDiagnoseFormat(shortBuffer, sizeof(shortBuffer), bufferArray, format, args...);
    getDebugCallback()->handleMessage(type, DebugMessageSource::Layer, buffer);
}

#define GFX_DIAGNOSE_ERROR(message)                                                                \
    _gfxDiagnoseImpl(                                                                              \
        DebugMessageType::Error,                                                                   \
        "%s: %s",                                                                                  \
        _gfxGetFuncName(_currentFunctionName ? _currentFunctionName : SLANG_FUNC_SIG).getBuffer(), \
        message)
#define GFX_DIAGNOSE_WARNING(message)                                                              \
    _gfxDiagnoseImpl(                                                                              \
        DebugMessageType::Warning,                                                                 \
        "%s: %s",                                                                                  \
        _gfxGetFuncName(_currentFunctionName ? _currentFunctionName : SLANG_FUNC_SIG).getBuffer(), \
        message)
#define GFX_DIAGNOSE_INFO(message)                                                                 \
    _gfxDiagnoseImpl(                                                                              \
        DebugMessageType::Info,                                                                    \
        "%s: %s",                                                                                  \
        _gfxGetFuncName(_currentFunctionName ? _currentFunctionName : SLANG_FUNC_SIG).getBuffer(), \
        message)
#define GFX_DIAGNOSE_FORMAT(type, format, ...)                                            \
    {                                                                                     \
        char shortBuffer[256];                                                            \
        List<char> bufferArray;                                                           \
        auto message = _gfxDiagnoseFormat(                                                \
            shortBuffer,                                                                  \
            sizeof(shortBuffer),                                                          \
            bufferArray,                                                                  \
            format,                                                                       \
            __VA_ARGS__);                                                                 \
        _gfxDiagnoseImpl(                                                                 \
            type,                                                                         \
            "%s: %s",                                                                     \
            _gfxGetFuncName(_currentFunctionName ? _currentFunctionName : SLANG_FUNC_SIG) \
                .getBuffer(),                                                             \
            message);                                                                     \
    }
#define GFX_DIAGNOSE_ERROR_FORMAT(...) GFX_DIAGNOSE_FORMAT(DebugMessageType::Error, __VA_ARGS__)

#define SLANG_GFX_DEBUG_GET_INTERFACE_IMPL(typeName)                                    \
    I##typeName* Debug##typeName::getInterface(const Slang::Guid& guid)                 \
    {                                                                                   \
        return (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_I##typeName) \
                   ? static_cast<I##typeName*>(this)                                    \
                   : nullptr;                                                           \
    }
#define SLANG_GFX_DEBUG_GET_INTERFACE_IMPL_PARENT(typeName, parentType)                   \
    I##typeName* Debug##typeName::getInterface(const Slang::Guid& guid)                   \
    {                                                                                     \
        return (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_I##typeName || \
                guid == GfxGUID::IID_I##parentType)                                       \
                   ? static_cast<I##typeName*>(this)                                      \
                   : nullptr;                                                             \
    }

// Utility conversion functions to get Debug* object or the inner object from a user provided
// pointer.
#define SLANG_GFX_DEBUG_GET_OBJ_IMPL(type)                                         \
    inline Debug##type* getDebugObj(I##type* ptr)                                  \
    {                                                                              \
        return static_cast<Debug##type*>(static_cast<DebugObject<I##type>*>(ptr)); \
    }                                                                              \
    inline I##type* getInnerObj(I##type* ptr)                                      \
    {                                                                              \
        if (!ptr)                                                                  \
            return nullptr;                                                        \
        auto debugObj = getDebugObj(ptr);                                          \
        return debugObj->baseObject;                                               \
    }

#define SLANG_GFX_DEBUG_GET_OBJ_IMPL_UNOWNED(type)                                        \
    inline Debug##type* getDebugObj(I##type* ptr)                                         \
    {                                                                                     \
        return static_cast<Debug##type*>(static_cast<UnownedDebugObject<I##type>*>(ptr)); \
    }                                                                                     \
    inline I##type* getInnerObj(I##type* ptr)                                             \
    {                                                                                     \
        if (!ptr)                                                                         \
            return nullptr;                                                               \
        auto debugObj = getDebugObj(ptr);                                                 \
        return debugObj->baseObject;                                                      \
    }

SLANG_GFX_DEBUG_GET_OBJ_IMPL(Device)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(BufferResource)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(TextureResource)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(CommandBuffer)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(CommandQueue)
SLANG_GFX_DEBUG_GET_OBJ_IMPL_UNOWNED(ComputeCommandEncoder)
SLANG_GFX_DEBUG_GET_OBJ_IMPL_UNOWNED(RenderCommandEncoder)
SLANG_GFX_DEBUG_GET_OBJ_IMPL_UNOWNED(ResourceCommandEncoder)
SLANG_GFX_DEBUG_GET_OBJ_IMPL_UNOWNED(RayTracingCommandEncoder)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(Framebuffer)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(FramebufferLayout)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(InputLayout)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(RenderPassLayout)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(PipelineState)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(ResourceView)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(SamplerState)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(ShaderObject)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(ShaderProgram)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(Swapchain)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(TransientResourceHeap)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(QueryPool)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(AccelerationStructure)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(Fence)
SLANG_GFX_DEBUG_GET_OBJ_IMPL(ShaderTable)

void validateAccelerationStructureBuildInputs(
    const IAccelerationStructure::BuildInputs& buildInputs);

} // namespace debug
} // namespace gfx
