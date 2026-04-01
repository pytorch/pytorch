#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

#if SLANG_WINDOWS_FAMILY
#include <d3d12.h>
#endif

using namespace Slang;
using namespace gfx;

namespace
{
using namespace gfx_test;

struct GetSupportedResourceStatesBase
{
    IDevice* device;
    UnitTestContext* context;

    ResourceStateSet formatSupportedStates;
    ResourceStateSet textureAllowedStates;
    ResourceStateSet bufferAllowedStates;

    ComPtr<ITextureResource> texture;
    ComPtr<IBufferResource> buffer;

    void init(IDevice* device, UnitTestContext* context)
    {
        this->device = device;
        this->context = context;
    }

    Format convertTypelessFormat(Format format)
    {
        switch (format)
        {
        case Format::R32G32B32A32_TYPELESS:
            return Format::R32G32B32A32_FLOAT;
        case Format::R32G32B32_TYPELESS:
            return Format::R32G32B32_FLOAT;
        case Format::R32G32_TYPELESS:
            return Format::R32G32_FLOAT;
        case Format::R32_TYPELESS:
            return Format::R32_FLOAT;
        case Format::R16G16B16A16_TYPELESS:
            return Format::R16G16B16A16_FLOAT;
        case Format::R16G16_TYPELESS:
            return Format::R16G16_FLOAT;
        case Format::R16_TYPELESS:
            return Format::R16_FLOAT;
        case Format::R8G8B8A8_TYPELESS:
            return Format::R8G8B8A8_UNORM;
        case Format::R8G8_TYPELESS:
            return Format::R8G8_UNORM;
        case Format::R8_TYPELESS:
            return Format::R8_UNORM;
        case Format::B8G8R8A8_TYPELESS:
            return Format::B8G8R8A8_UNORM;
        case Format::R10G10B10A2_TYPELESS:
            return Format::R10G10B10A2_UINT;
        default:
            return Format::Unknown;
        }
    }

    void transitionResourceStates(IDevice* device)
    {
        Slang::ComPtr<ITransientResourceHeap> transientHeap;
        ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096;
        GFX_CHECK_CALL_ABORT(
            device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = device->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        auto encoder = commandBuffer->encodeResourceCommands();
        ResourceState currentTextureState = texture->getDesc()->defaultState;
        ResourceState currentBufferState = buffer->getDesc()->defaultState;

        for (uint32_t i = 0; i < (uint32_t)ResourceState::_Count; ++i)
        {
            auto nextState = (ResourceState)i;
            if (formatSupportedStates.contains(nextState))
            {
                if (bufferAllowedStates.contains(nextState))
                {
                    encoder->bufferBarrier(buffer, currentBufferState, nextState);
                    currentBufferState = nextState;
                }
                if (textureAllowedStates.contains(nextState))
                {
                    encoder->textureBarrier(texture, currentTextureState, nextState);
                    currentTextureState = nextState;
                }
            }
        }
        encoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    void run()
    {
        // Skip Format::Unknown
        for (uint32_t i = 1; i < (uint32_t)Format::_Count; ++i)
        {
            auto baseFormat = (Format)i;
            FormatInfo info;
            gfxGetFormatInfo(baseFormat, &info);
            // Ignore 3-channel textures for now since validation layer seem to report unsupported
            // errors there.
            if (info.channelCount == 3)
                continue;

            auto format =
                gfxIsTypelessFormat(baseFormat) ? convertTypelessFormat(baseFormat) : baseFormat;
            GFX_CHECK_CALL_ABORT(
                device->getFormatSupportedResourceStates(format, &formatSupportedStates));

            textureAllowedStates.add(
                ResourceState::RenderTarget,
                ResourceState::DepthRead,
                ResourceState::DepthWrite,
                ResourceState::Present,
                ResourceState::ResolveSource,
                ResourceState::ResolveDestination,
                ResourceState::Undefined,
                ResourceState::ShaderResource,
                ResourceState::UnorderedAccess,
                ResourceState::CopySource,
                ResourceState::CopyDestination);

            bufferAllowedStates.add(
                ResourceState::VertexBuffer,
                ResourceState::IndexBuffer,
                ResourceState::ConstantBuffer,
                ResourceState::StreamOutput,
                ResourceState::IndirectArgument,
                ResourceState::AccelerationStructure,
                ResourceState::Undefined,
                ResourceState::ShaderResource,
                ResourceState::UnorderedAccess,
                ResourceState::CopySource,
                ResourceState::CopyDestination);

            ResourceState currentState = ResourceState::CopySource;
            ITextureResource::Extents extent;
            extent.width = 4;
            extent.height = 4;
            extent.depth = 1;

            ITextureResource::Desc texDesc = {};
            texDesc.type = IResource::Type::Texture2D;
            texDesc.numMipLevels = 1;
            texDesc.arraySize = 1;
            texDesc.size = extent;
            texDesc.defaultState = currentState;
            texDesc.allowedStates = formatSupportedStates & textureAllowedStates;
            texDesc.memoryType = MemoryType::DeviceLocal;
            texDesc.format = format;

            GFX_CHECK_CALL_ABORT(
                device->createTextureResource(texDesc, nullptr, texture.writeRef()));

            IBufferResource::Desc bufferDesc = {};
            bufferDesc.sizeInBytes = 256;
            bufferDesc.format = gfx::Format::Unknown;
            bufferDesc.elementSize = sizeof(float);
            bufferDesc.allowedStates = formatSupportedStates & bufferAllowedStates;
            bufferDesc.defaultState = currentState;
            bufferDesc.memoryType = MemoryType::DeviceLocal;

            GFX_CHECK_CALL_ABORT(
                device->createBufferResource(bufferDesc, nullptr, buffer.writeRef()));

            transitionResourceStates(device);
        }
    }
};

void supportedResourceStatesTestImpl(IDevice* device, UnitTestContext* context)
{
    GetSupportedResourceStatesBase test;
    test.init(device, context);
    test.run();
}
} // namespace

namespace gfx_test
{
SLANG_UNIT_TEST(getSupportedResourceStatesD3D12)
{
    runTestImpl(supportedResourceStatesTestImpl, unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(getSupportedResourceStatesVulkan)
{
    runTestImpl(supportedResourceStatesTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}
} // namespace gfx_test
