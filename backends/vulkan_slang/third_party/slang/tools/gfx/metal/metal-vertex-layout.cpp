// metal-vertex-layout.cpp
#include "metal-vertex-layout.h"

#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

Result InputLayoutImpl::init(const IInputLayout::Desc& desc)
{
    for (Index i = 0; i < desc.inputElementCount; i++)
    {
        if (MetalUtil::translateVertexFormat(desc.inputElements[i].format) ==
            MTL::VertexFormatInvalid)
        {
            return SLANG_E_INVALID_ARG;
        }
        m_inputElements.add(desc.inputElements[i]);
    }
    for (Index i = 0; i < desc.vertexStreamCount; i++)
    {
        m_vertexStreams.add(desc.vertexStreams[i]);
    }
    return SLANG_OK;
}

NS::SharedPtr<MTL::VertexDescriptor> InputLayoutImpl::createVertexDescriptor(
    NS::UInteger vertexBufferIndexOffset)
{
    NS::SharedPtr<MTL::VertexDescriptor> vertexDescriptor =
        NS::TransferPtr(MTL::VertexDescriptor::alloc()->init());

    for (Index i = 0; i < m_inputElements.getCount(); i++)
    {
        const auto& inputElement = m_inputElements[i];
        MTL::VertexAttributeDescriptor* desc = vertexDescriptor->attributes()->object(i);
        desc->setOffset(inputElement.offset);
        desc->setBufferIndex(inputElement.bufferSlotIndex + vertexBufferIndexOffset);
        MTL::VertexFormat metalFormat = MetalUtil::translateVertexFormat(inputElement.format);
        desc->setFormat(metalFormat);
    }

    for (Index i = 0; i < m_vertexStreams.getCount(); i++)
    {
        const auto& vertexStream = m_vertexStreams[i];
        MTL::VertexBufferLayoutDescriptor* desc =
            vertexDescriptor->layouts()->object(i + vertexBufferIndexOffset);
        desc->setStepFunction(MetalUtil::translateVertexStepFunction(vertexStream.slotClass));
        desc->setStepRate(
            vertexStream.slotClass == InputSlotClass::PerVertex
                ? 1
                : vertexStream.instanceDataStepRate);
        desc->setStride(vertexStream.stride);
    }

    return vertexDescriptor;
}

} // namespace metal
} // namespace gfx
