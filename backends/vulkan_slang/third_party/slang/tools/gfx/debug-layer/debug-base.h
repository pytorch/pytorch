// debug-base.h
#pragma once

#include "../command-encoder-com-forward.h"
#include "../renderer-shared.h"
#include "core/slang-com-object.h"
#include "slang-com-ptr.h"
#include "slang-gfx.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugObjectBase : public Slang::ComObject
{
public:
    uint64_t uid;
    DebugObjectBase()
    {
        static uint64_t uidCounter = 0;
        uid = ++uidCounter;
    }
};

template<typename TInterface>
class DebugObject : public TInterface, public DebugObjectBase
{
public:
    Slang::ComPtr<TInterface> baseObject;
};

template<typename TInterface>
class UnownedDebugObject : public TInterface, public DebugObjectBase
{
public:
    TInterface* baseObject = nullptr;
};

class DebugDevice;
class DebugShaderTable;
class DebugQueryPool;
class DebugBufferResource;
class DebugTextureResource;
class DebugResourceView;
class DebugAccelerationStructure;
class DebugSamplerState;
class DebugShaderObject;
class DebugRootShaderObject;
class DebugCommandBuffer;
class DebugResourceCommandEncoderImpl;
class DebugComputeCommandEncoder;
class DebugResourceCommandEncoder;
class DebugRenderCommandEncoder;
class DebugRayTracingCommandEncoder;
class DebugFence;
class DebugCommandQueue;
class DebugFramebuffer;
class DebugFramebufferLayout;
class DebugInputLayout;
class DebugPipelineState;
class DebugRenderPassLayout;
class DebugShaderProgram;
class DebugTransientResourceHeap;
class DebugSwapchain;

} // namespace debug
} // namespace gfx
