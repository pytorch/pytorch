// d3d12-shader-table.cpp
#include "d3d12-shader-table.h"

#include "d3d12-device.h"
#include "d3d12-pipeline-state.h"
#include "d3d12-transient-heap.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

RefPtr<BufferResource> ShaderTableImpl::createDeviceBuffer(
    PipelineStateBase* pipeline,
    TransientResourceHeapBase* transientHeap,
    IResourceCommandEncoder* encoder)
{
    uint32_t raygenTableSize = m_rayGenShaderCount * kRayGenRecordSize;
    uint32_t missTableSize = m_missShaderCount * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    uint32_t hitgroupTableSize = m_hitGroupCount * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    uint32_t callableTableSize = m_callableShaderCount * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    m_rayGenTableOffset = 0;
    m_missTableOffset = raygenTableSize;
    m_hitGroupTableOffset = (uint32_t)D3DUtil::calcAligned(
        m_missTableOffset + missTableSize,
        D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    m_callableTableOffset = (uint32_t)D3DUtil::calcAligned(
        m_hitGroupTableOffset + hitgroupTableSize,
        D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    uint32_t tableSize = m_callableTableOffset + callableTableSize;

    auto pipelineImpl = static_cast<RayTracingPipelineStateImpl*>(pipeline);
    ComPtr<IBufferResource> bufferResource;
    IBufferResource::Desc bufferDesc = {};
    bufferDesc.memoryType = gfx::MemoryType::DeviceLocal;
    bufferDesc.defaultState = ResourceState::General;
    bufferDesc.allowedStates.add(ResourceState::NonPixelShaderResource);
    bufferDesc.type = IResource::Type::Buffer;
    bufferDesc.sizeInBytes = tableSize;
    m_device->createBufferResource(bufferDesc, nullptr, bufferResource.writeRef());

    ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
    pipelineImpl->m_stateObject->QueryInterface(stateObjectProperties.writeRef());

    TransientResourceHeapImpl* transientHeapImpl =
        static_cast<TransientResourceHeapImpl*>(transientHeap);

    IBufferResource* stagingBuffer = nullptr;
    Offset stagingBufferOffset = 0;
    transientHeapImpl
        ->allocateStagingBuffer(tableSize, stagingBuffer, stagingBufferOffset, MemoryType::Upload);

    assert(stagingBuffer);
    void* stagingPtr = nullptr;
    stagingBuffer->map(nullptr, &stagingPtr);

    auto copyShaderIdInto = [&](void* dest, String& name, const ShaderRecordOverwrite& overwrite)
    {
        if (name.getLength())
        {
            void* shaderId = stateObjectProperties->GetShaderIdentifier(name.toWString().begin());
            if (nullptr == shaderId)
                throw Exception(String("Failed to get shader identifier for '") + name + "'");
            memcpy(dest, shaderId, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        }
        if (overwrite.size)
        {
            memcpy((uint8_t*)dest + overwrite.offset, overwrite.data, overwrite.size);
        }
    };

    uint8_t* stagingBufferPtr = (uint8_t*)stagingPtr + stagingBufferOffset;
    memset(stagingBufferPtr, 0, tableSize);

    for (uint32_t i = 0; i < m_rayGenShaderCount; i++)
    {
        copyShaderIdInto(
            stagingBufferPtr + m_rayGenTableOffset + kRayGenRecordSize * i,
            m_shaderGroupNames[i],
            m_recordOverwrites[i]);
    }
    for (uint32_t i = 0; i < m_missShaderCount; i++)
    {
        copyShaderIdInto(
            stagingBufferPtr + m_missTableOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * i,
            m_shaderGroupNames[m_rayGenShaderCount + i],
            m_recordOverwrites[m_rayGenShaderCount + i]);
    }
    for (uint32_t i = 0; i < m_hitGroupCount; i++)
    {
        copyShaderIdInto(
            stagingBufferPtr + m_hitGroupTableOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * i,
            m_shaderGroupNames[m_rayGenShaderCount + m_missShaderCount + i],
            m_recordOverwrites[m_rayGenShaderCount + m_missShaderCount + i]);
    }
    for (uint32_t i = 0; i < m_callableShaderCount; i++)
    {
        copyShaderIdInto(
            stagingBufferPtr + m_callableTableOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * i,
            m_shaderGroupNames[m_rayGenShaderCount + m_missShaderCount + m_hitGroupCount + i],
            m_recordOverwrites[m_rayGenShaderCount + m_missShaderCount + m_hitGroupCount + i]);
    }

    stagingBuffer->unmap(nullptr);
    encoder->copyBuffer(bufferResource, 0, stagingBuffer, stagingBufferOffset, tableSize);
    encoder->bufferBarrier(
        1,
        bufferResource.readRef(),
        gfx::ResourceState::CopyDestination,
        gfx::ResourceState::NonPixelShaderResource);
    RefPtr<BufferResource> resultPtr = static_cast<BufferResource*>(bufferResource.get());
    return _Move(resultPtr);
}

} // namespace d3d12
} // namespace gfx
