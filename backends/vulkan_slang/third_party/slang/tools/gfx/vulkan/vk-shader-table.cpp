// vk-shader-table.cpp
#include "vk-shader-table.h"

#include "vk-device.h"
#include "vk-helper-functions.h"
#include "vk-transient-heap.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

RefPtr<BufferResource> ShaderTableImpl::createDeviceBuffer(
    PipelineStateBase* pipeline,
    TransientResourceHeapBase* transientHeap,
    IResourceCommandEncoder* encoder)
{
    auto vkApi = m_device->m_api;
    auto rtProps = vkApi.m_rtProperties;
    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    m_raygenTableSize = m_rayGenShaderCount * rtProps.shaderGroupBaseAlignment;
    m_missTableSize = (uint32_t)VulkanUtil::calcAligned(
        m_missShaderCount * handleSize,
        rtProps.shaderGroupBaseAlignment);
    m_hitTableSize = (uint32_t)VulkanUtil::calcAligned(
        m_hitGroupCount * handleSize,
        rtProps.shaderGroupBaseAlignment);
    m_callableTableSize = (uint32_t)VulkanUtil::calcAligned(
        m_callableShaderCount * handleSize,
        rtProps.shaderGroupBaseAlignment);
    uint32_t tableSize = m_raygenTableSize + m_missTableSize + m_hitTableSize + m_callableTableSize;

    auto pipelineImpl = static_cast<RayTracingPipelineStateImpl*>(pipeline);
    ComPtr<IBufferResource> bufferResource;
    IBufferResource::Desc bufferDesc = {};
    bufferDesc.memoryType = MemoryType::DeviceLocal;
    bufferDesc.defaultState = ResourceState::General;
    bufferDesc.allowedStates =
        ResourceStateSet(ResourceState::General, ResourceState::CopyDestination);
    bufferDesc.type = IResource::Type::Buffer;
    bufferDesc.sizeInBytes = tableSize;
    static_cast<vk::DeviceImpl*>(m_device)->createBufferResourceImpl(
        bufferDesc,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
        nullptr,
        bufferResource.writeRef());

    TransientResourceHeapImpl* transientHeapImpl =
        static_cast<TransientResourceHeapImpl*>(transientHeap);

    IBufferResource* stagingBuffer = nullptr;
    Offset stagingBufferOffset = 0;
    transientHeapImpl
        ->allocateStagingBuffer(tableSize, stagingBuffer, stagingBufferOffset, MemoryType::Upload);

    assert(stagingBuffer);
    void* stagingPtr = nullptr;
    stagingBuffer->map(nullptr, &stagingPtr);

    List<uint8_t> handles;
    auto handleCount = pipelineImpl->shaderGroupCount;
    auto totalHandleSize = handleSize * handleCount;
    handles.setCount(totalHandleSize);
    auto result = vkApi.vkGetRayTracingShaderGroupHandlesKHR(
        m_device->m_device,
        pipelineImpl->m_pipeline,
        0,
        (uint32_t)handleCount,
        totalHandleSize,
        handles.getBuffer());

    uint8_t* stagingBufferPtr = (uint8_t*)stagingPtr + stagingBufferOffset;
    auto subTablePtr = stagingBufferPtr;
    Int shaderTableEntryCounter = 0;

    // Each loop calculates the copy source and destination locations by fetching the name
    // of the shader group from the list of shader group names and getting its corresponding
    // index in the buffer of handles.
    for (uint32_t i = 0; i < m_rayGenShaderCount; i++)
    {
        auto dstHandlePtr = subTablePtr + i * rtProps.shaderGroupBaseAlignment;
        auto shaderGroupName = m_shaderGroupNames[shaderTableEntryCounter++];
        auto shaderGroupIndexPtr =
            pipelineImpl->shaderGroupNameToIndex.tryGetValue(shaderGroupName);
        if (!shaderGroupIndexPtr)
            continue;

        auto shaderGroupIndex = *shaderGroupIndexPtr;
        auto srcHandlePtr = handles.getBuffer() + shaderGroupIndex * handleSize;
        memcpy(dstHandlePtr, srcHandlePtr, handleSize);
        memset(dstHandlePtr + handleSize, 0, rtProps.shaderGroupBaseAlignment - handleSize);
    }
    subTablePtr += m_raygenTableSize;

    for (uint32_t i = 0; i < m_missShaderCount; i++)
    {
        auto dstHandlePtr = subTablePtr + i * handleSize;
        auto shaderGroupName = m_shaderGroupNames[shaderTableEntryCounter++];
        auto shaderGroupIndexPtr =
            pipelineImpl->shaderGroupNameToIndex.tryGetValue(shaderGroupName);
        if (!shaderGroupIndexPtr)
            continue;

        auto shaderGroupIndex = *shaderGroupIndexPtr;
        auto srcHandlePtr = handles.getBuffer() + shaderGroupIndex * handleSize;
        memcpy(dstHandlePtr, srcHandlePtr, handleSize);
    }
    subTablePtr += m_missTableSize;

    for (uint32_t i = 0; i < m_hitGroupCount; i++)
    {
        auto dstHandlePtr = subTablePtr + i * handleSize;
        auto shaderGroupName = m_shaderGroupNames[shaderTableEntryCounter++];
        auto shaderGroupIndexPtr =
            pipelineImpl->shaderGroupNameToIndex.tryGetValue(shaderGroupName);
        if (!shaderGroupIndexPtr)
            continue;

        auto shaderGroupIndex = *shaderGroupIndexPtr;
        auto srcHandlePtr = handles.getBuffer() + shaderGroupIndex * handleSize;
        memcpy(dstHandlePtr, srcHandlePtr, handleSize);
    }
    subTablePtr += m_hitTableSize;

    for (uint32_t i = 0; i < m_callableShaderCount; i++)
    {
        auto dstHandlePtr = subTablePtr + i * handleSize;
        auto shaderGroupName = m_shaderGroupNames[shaderTableEntryCounter++];
        auto shaderGroupIndexPtr =
            pipelineImpl->shaderGroupNameToIndex.tryGetValue(shaderGroupName);
        if (!shaderGroupIndexPtr)
            continue;

        auto shaderGroupIndex = *shaderGroupIndexPtr;
        auto srcHandlePtr = handles.getBuffer() + shaderGroupIndex * handleSize;
        memcpy(dstHandlePtr, srcHandlePtr, handleSize);
    }
    subTablePtr += m_callableTableSize;

    stagingBuffer->unmap(nullptr);
    encoder->copyBuffer(bufferResource, 0, stagingBuffer, stagingBufferOffset, tableSize);
    encoder->bufferBarrier(
        1,
        bufferResource.readRef(),
        gfx::ResourceState::CopyDestination,
        gfx::ResourceState::ShaderResource);
    RefPtr<BufferResource> resultPtr = static_cast<BufferResource*>(bufferResource.get());
    return _Move(resultPtr);
}

} // namespace vk
} // namespace gfx
