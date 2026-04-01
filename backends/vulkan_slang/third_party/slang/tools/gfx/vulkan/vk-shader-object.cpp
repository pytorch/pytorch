// vk-shader-object.cpp
#include "vk-shader-object.h"

#include "vk-command-buffer.h"
#include "vk-command-encoder.h"
#include "vk-transient-heap.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

Result ShaderObjectImpl::create(
    IDevice* device,
    ShaderObjectLayoutImpl* layout,
    ShaderObjectImpl** outShaderObject)
{
    auto object = RefPtr<ShaderObjectImpl>(new ShaderObjectImpl());
    SLANG_RETURN_ON_FAIL(object->init(device, layout));

    returnRefPtrMove(outShaderObject, object);
    return SLANG_OK;
}

RendererBase* ShaderObjectImpl::getDevice()
{
    return m_layout->getDevice();
}

GfxCount ShaderObjectImpl::getEntryPointCount()
{
    return 0;
}

Result ShaderObjectImpl::getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint)
{
    *outEntryPoint = nullptr;
    return SLANG_OK;
}

const void* ShaderObjectImpl::getRawData()
{
    return m_data.getBuffer();
}

Size ShaderObjectImpl::getSize()
{
    return (Size)m_data.getCount();
}

// TODO: Change size_t and Index to Size?
Result ShaderObjectImpl::setData(ShaderOffset const& inOffset, void const* data, size_t inSize)
{
    Index offset = inOffset.uniformOffset;
    Index size = inSize;

    char* dest = m_data.getBuffer();
    Index availableSize = m_data.getCount();

    // TODO: We really should bounds-check access rather than silently ignoring sets
    // that are too large, but we have several test cases that set more data than
    // an object actually stores on several targets...
    //
    if (offset < 0)
    {
        size += offset;
        offset = 0;
    }
    if ((offset + size) >= availableSize)
    {
        size = availableSize - offset;
    }

    memcpy(dest + offset, data, size);

    m_isConstantBufferDirty = true;

    return SLANG_OK;
}

Result ShaderObjectImpl::setResource(ShaderOffset const& offset, IResourceView* resourceView)
{
    if (offset.bindingRangeIndex < 0)
        return SLANG_E_INVALID_ARG;
    auto layout = getLayout();
    if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
        return SLANG_E_INVALID_ARG;
    auto& bindingRange = layout->getBindingRange(offset.bindingRangeIndex);
    if (!resourceView)
    {
        m_resourceViews[bindingRange.baseIndex + offset.bindingArrayIndex] = nullptr;
    }
    else
    {
        if (resourceView->getViewDesc()->type == IResourceView::Type::AccelerationStructure)
        {
            m_resourceViews[bindingRange.baseIndex + offset.bindingArrayIndex] =
                static_cast<AccelerationStructureImpl*>(resourceView);
        }
        else
        {
            m_resourceViews[bindingRange.baseIndex + offset.bindingArrayIndex] =
                static_cast<ResourceViewImpl*>(resourceView);
        }
    }
    return SLANG_OK;
}

Result ShaderObjectImpl::setSampler(ShaderOffset const& offset, ISamplerState* sampler)
{
    if (offset.bindingRangeIndex < 0)
        return SLANG_E_INVALID_ARG;
    auto layout = getLayout();
    if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
        return SLANG_E_INVALID_ARG;
    auto& bindingRange = layout->getBindingRange(offset.bindingRangeIndex);

    m_samplers[bindingRange.baseIndex + offset.bindingArrayIndex] =
        static_cast<SamplerStateImpl*>(sampler);
    return SLANG_OK;
}

Result ShaderObjectImpl::setCombinedTextureSampler(
    ShaderOffset const& offset,
    IResourceView* textureView,
    ISamplerState* sampler)
{
    if (offset.bindingRangeIndex < 0)
        return SLANG_E_INVALID_ARG;
    auto layout = getLayout();
    if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
        return SLANG_E_INVALID_ARG;
    auto& bindingRange = layout->getBindingRange(offset.bindingRangeIndex);

    auto& slot = m_combinedTextureSamplers[bindingRange.baseIndex + offset.bindingArrayIndex];
    slot.textureView = static_cast<TextureResourceViewImpl*>(textureView);
    slot.sampler = static_cast<SamplerStateImpl*>(sampler);
    return SLANG_OK;
}

Result ShaderObjectImpl::init(IDevice* device, ShaderObjectLayoutImpl* layout)
{
    m_layout = layout;

    m_constantBufferTransientHeap = nullptr;
    m_constantBufferTransientHeapVersion = 0;
    m_isConstantBufferDirty = true;

    // If the layout tells us that there is any uniform data,
    // then we will allocate a CPU memory buffer to hold that data
    // while it is being set from the host.
    //
    // Once the user is done setting the parameters/fields of this
    // shader object, we will produce a GPU-memory version of the
    // uniform data (which includes values from this object and
    // any existential-type sub-objects).
    //
    // TODO: Change size_t to Count?
    size_t uniformSize = layout->getElementTypeLayout()->getSize();
    if (uniformSize)
    {
        m_data.setCount(uniformSize);
        memset(m_data.getBuffer(), 0, uniformSize);
    }

#if 0
        // If the layout tells us there are any descriptor sets to
        // allocate, then we do so now.
        //
        for(auto descriptorSetInfo : layout->getDescriptorSets())
        {
            RefPtr<DescriptorSet> descriptorSet;
            SLANG_RETURN_ON_FAIL(renderer->createDescriptorSet(descriptorSetInfo->layout, descriptorSet.writeRef()));
            m_descriptorSets.add(descriptorSet);
        }
#endif

    m_resourceViews.setCount(layout->getResourceViewCount());
    m_samplers.setCount(layout->getSamplerCount());
    m_combinedTextureSamplers.setCount(layout->getCombinedTextureSamplerCount());

    // If the layout specifies that we have any sub-objects, then
    // we need to size the array to account for them.
    //
    Index subObjectCount = layout->getSubObjectCount();
    m_objects.setCount(subObjectCount);

    for (auto subObjectRangeInfo : layout->getSubObjectRanges())
    {
        auto subObjectLayout = subObjectRangeInfo.layout;

        // In the case where the sub-object range represents an
        // existential-type leaf field (e.g., an `IBar`), we
        // cannot pre-allocate the object(s) to go into that
        // range, since we can't possibly know what to allocate
        // at this point.
        //
        if (!subObjectLayout)
            continue;
        //
        // Otherwise, we will allocate a sub-object to fill
        // in each entry in this range, based on the layout
        // information we already have.

        auto& bindingRangeInfo = layout->getBindingRange(subObjectRangeInfo.bindingRangeIndex);
        for (Index i = 0; i < bindingRangeInfo.count; ++i)
        {
            RefPtr<ShaderObjectImpl> subObject;
            SLANG_RETURN_ON_FAIL(
                ShaderObjectImpl::create(device, subObjectLayout, subObject.writeRef()));
            m_objects[bindingRangeInfo.subObjectIndex + i] = subObject;
        }
    }

    return SLANG_OK;
}

Result ShaderObjectImpl::_writeOrdinaryData(
    PipelineCommandEncoder* encoder,
    IBufferResource* buffer,
    Offset offset,
    Size destSize,
    ShaderObjectLayoutImpl* specializedLayout)
{
    auto src = m_data.getBuffer();
    // TODO: Change size_t to Count?
    auto srcSize = size_t(m_data.getCount());

    SLANG_ASSERT(srcSize <= destSize);

    encoder->uploadBufferDataImpl(buffer, offset, srcSize, src);

    // In the case where this object has any sub-objects of
    // existential/interface type, we need to recurse on those objects
    // that need to write their state into an appropriate "pending" allocation.
    //
    // Note: Any values that could fit into the "payload" included
    // in the existential-type field itself will have already been
    // written as part of `setObject()`. This loop only needs to handle
    // those sub-objects that do not "fit."
    //
    // An implementers looking at this code might wonder if things could be changed
    // so that *all* writes related to sub-objects for interface-type fields could
    // be handled in this one location, rather than having some in `setObject()` and
    // others handled here.
    //
    Index subObjectRangeCounter = 0;
    for (auto const& subObjectRangeInfo : specializedLayout->getSubObjectRanges())
    {
        Index subObjectRangeIndex = subObjectRangeCounter++;
        auto const& bindingRangeInfo =
            specializedLayout->getBindingRange(subObjectRangeInfo.bindingRangeIndex);

        // We only need to handle sub-object ranges for interface/existential-type fields,
        // because fields of constant-buffer or parameter-block type are responsible for
        // the ordinary/uniform data of their own existential/interface-type sub-objects.
        //
        if (bindingRangeInfo.bindingType != slang::BindingType::ExistentialValue)
            continue;

        // Each sub-object range represents a single "leaf" field, but might be nested
        // under zero or more outer arrays, such that the number of existential values
        // in the same range can be one or more.
        //
        auto count = bindingRangeInfo.count;

        // We are not concerned with the case where the existential value(s) in the range
        // git into the payload part of the leaf field.
        //
        // In the case where the value didn't fit, the Slang layout strategy would have
        // considered the requirements of the value as a "pending" allocation, and would
        // allocate storage for the ordinary/uniform part of that pending allocation inside
        // of the parent object's type layout.
        //
        // Here we assume that the Slang reflection API can provide us with a single byte
        // offset and stride for the location of the pending data allocation in the
        // specialized type layout, which will store the values for this sub-object range.
        //
        // TODO: The reflection API functions we are assuming here haven't been implemented
        // yet, so the functions being called here are stubs.
        //
        // TODO: It might not be that a single sub-object range can reliably map to a single
        // contiguous array with a single stride; we need to carefully consider what the
        // layout logic does for complex cases with multiple layers of nested arrays and
        // structures.
        //
        Offset subObjectRangePendingDataOffset = subObjectRangeInfo.offset.pendingOrdinaryData;
        Size subObjectRangePendingDataStride = subObjectRangeInfo.stride.pendingOrdinaryData;

        // If the range doesn't actually need/use the "pending" allocation at all, then
        // we need to detect that case and skip such ranges.
        //
        // TODO: This should probably be handled on a per-object basis by caching a "does it
        // fit?" bit as part of the information for bound sub-objects, given that we already
        // compute the "does it fit?" status as part of `setObject()`.
        //
        if (subObjectRangePendingDataOffset == 0)
            continue;

        for (Slang::Index i = 0; i < count; ++i)
        {
            auto subObject = m_objects[bindingRangeInfo.subObjectIndex + i];

            RefPtr<ShaderObjectLayoutImpl> subObjectLayout;
            SLANG_RETURN_ON_FAIL(subObject->_getSpecializedLayout(subObjectLayout.writeRef()));

            auto subObjectOffset =
                subObjectRangePendingDataOffset + i * subObjectRangePendingDataStride;

            subObject->_writeOrdinaryData(
                encoder,
                buffer,
                offset + subObjectOffset,
                destSize - subObjectOffset,
                subObjectLayout);
        }
    }

    return SLANG_OK;
}

void ShaderObjectImpl::writeDescriptor(
    RootBindingContext& context,
    VkWriteDescriptorSet const& write)
{
    auto device = context.device;
    device->m_api.vkUpdateDescriptorSets(device->m_device, 1, &write, 0, nullptr);
}

void ShaderObjectImpl::writeBufferDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    BufferResourceImpl* buffer,
    Offset bufferOffset,
    Size bufferSize)
{
    auto descriptorSet = (*context.descriptorSets)[offset.bindingSet];

    VkDescriptorBufferInfo bufferInfo = {};
    if (buffer)
    {
        bufferInfo.buffer = buffer->m_buffer.m_buffer;
    }
    bufferInfo.offset = bufferOffset;
    bufferInfo.range = bufferSize;

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.descriptorCount = 1;
    write.descriptorType = descriptorType;
    write.dstArrayElement = 0;
    write.dstBinding = offset.binding;
    write.dstSet = descriptorSet;
    write.pBufferInfo = &bufferInfo;

    writeDescriptor(context, write);
}

void ShaderObjectImpl::writeBufferDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    BufferResourceImpl* buffer)
{
    writeBufferDescriptor(
        context,
        offset,
        descriptorType,
        buffer,
        0,
        buffer->getDesc()->sizeInBytes);
}

void ShaderObjectImpl::writePlainBufferDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    ArrayView<RefPtr<ResourceViewInternalBase>> resourceViews)
{
    auto descriptorSet = (*context.descriptorSets)[offset.bindingSet];

    Index count = resourceViews.getCount();
    for (Index i = 0; i < count; ++i)
    {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.range = VK_WHOLE_SIZE;

        if (resourceViews[i])
        {
            auto boundViewType = static_cast<ResourceViewImpl*>(resourceViews[i].Ptr())->m_type;
            if (boundViewType == ResourceViewImpl::ViewType::PlainBuffer)
            {
                auto bufferView = static_cast<PlainBufferResourceViewImpl*>(resourceViews[i].Ptr());
                bufferInfo.buffer = bufferView->m_buffer->m_buffer.m_buffer;
                bufferInfo.offset = bufferView->offset;
                bufferInfo.range = bufferView->size;
            }
        }

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType;
        write.dstArrayElement = uint32_t(i);
        write.dstBinding = offset.binding;
        write.dstSet = descriptorSet;
        write.pBufferInfo = &bufferInfo;

        writeDescriptor(context, write);
    }
}

void ShaderObjectImpl::writeTexelBufferDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    ArrayView<RefPtr<ResourceViewInternalBase>> resourceViews)
{
    auto descriptorSet = (*context.descriptorSets)[offset.bindingSet];

    Index count = resourceViews.getCount();
    for (Index i = 0; i < count; ++i)
    {
        VkBufferView bufferView = VK_NULL_HANDLE;
        if (resourceViews[i])
        {
            auto boundViewType = static_cast<ResourceViewImpl*>(resourceViews[i].Ptr())->m_type;
            if (boundViewType == ResourceViewImpl::ViewType::TexelBuffer)
            {
                auto resourceView =
                    static_cast<TexelBufferResourceViewImpl*>(resourceViews[i].Ptr());
                bufferView = resourceView->m_view;
            }
        }
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorType = descriptorType;
        write.dstArrayElement = uint32_t(i);
        write.dstBinding = offset.binding;
        write.dstSet = descriptorSet;
        write.descriptorCount = 1;
        write.pTexelBufferView = &bufferView;
        writeDescriptor(context, write);
    }
}

void ShaderObjectImpl::writeTextureSamplerDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    ArrayView<CombinedTextureSamplerSlot> slots)
{
    auto descriptorSet = (*context.descriptorSets)[offset.bindingSet];

    Index count = slots.getCount();
    for (Index i = 0; i < count; ++i)
    {
        auto texture = slots[i].textureView;
        auto sampler = slots[i].sampler;
        VkDescriptorImageInfo imageInfo = {};
        if (texture)
        {
            imageInfo.imageView = texture->m_view;
            imageInfo.imageLayout = texture->m_layout;
        }
        if (sampler)
        {
            imageInfo.sampler = sampler->m_sampler;
        }
        else
        {
            imageInfo.sampler = context.device->m_defaultSampler;
        }

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType;
        write.dstArrayElement = uint32_t(i);
        write.dstBinding = offset.binding;
        write.dstSet = descriptorSet;
        write.pImageInfo = &imageInfo;

        writeDescriptor(context, write);
    }
}

void ShaderObjectImpl::writeAccelerationStructureDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    ArrayView<RefPtr<ResourceViewInternalBase>> resourceViews)
{
    auto descriptorSet = (*context.descriptorSets)[offset.bindingSet];

    Index count = resourceViews.getCount();
    for (Index i = 0; i < count; ++i)
    {
        auto accelerationStructure =
            static_cast<AccelerationStructureImpl*>(resourceViews[i].Ptr());
        VkWriteDescriptorSetAccelerationStructureKHR writeAS = {};
        writeAS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        VkAccelerationStructureKHR nullHandle = VK_NULL_HANDLE;
        if (accelerationStructure)
        {
            writeAS.accelerationStructureCount = 1;
            writeAS.pAccelerationStructures = &accelerationStructure->m_vkHandle;
        }
        else
        {
            writeAS.accelerationStructureCount = 1;
            writeAS.pAccelerationStructures = &nullHandle;
        }
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType;
        write.dstArrayElement = uint32_t(i);
        write.dstBinding = offset.binding;
        write.dstSet = descriptorSet;
        write.pNext = &writeAS;
        writeDescriptor(context, write);
    }
}

void ShaderObjectImpl::writeTextureDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    ArrayView<RefPtr<ResourceViewInternalBase>> resourceViews)
{
    auto descriptorSet = (*context.descriptorSets)[offset.bindingSet];

    Index count = resourceViews.getCount();
    for (Index i = 0; i < count; ++i)
    {
        VkDescriptorImageInfo imageInfo = {};
        if (resourceViews[i])
        {
            auto boundViewType = static_cast<ResourceViewImpl*>(resourceViews[i].Ptr())->m_type;
            if (boundViewType == ResourceViewImpl::ViewType::Texture)
            {
                auto texture = static_cast<TextureResourceViewImpl*>(resourceViews[i].Ptr());
                imageInfo.imageView = texture->m_view;
                imageInfo.imageLayout = texture->m_layout;
            }
        }
        imageInfo.sampler = 0;

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType;
        write.dstArrayElement = uint32_t(i);
        write.dstBinding = offset.binding;
        write.dstSet = descriptorSet;
        write.pImageInfo = &imageInfo;

        writeDescriptor(context, write);
    }
}

void ShaderObjectImpl::writeSamplerDescriptor(
    RootBindingContext& context,
    BindingOffset const& offset,
    VkDescriptorType descriptorType,
    ArrayView<RefPtr<SamplerStateImpl>> samplers)
{
    auto descriptorSet = (*context.descriptorSets)[offset.bindingSet];

    Index count = samplers.getCount();
    for (Index i = 0; i < count; ++i)
    {
        auto sampler = samplers[i];
        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageView = 0;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        if (sampler)
        {
            imageInfo.sampler = sampler->m_sampler;
        }
        else
        {
            imageInfo.sampler = context.device->m_defaultSampler;
        }

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType;
        write.dstArrayElement = uint32_t(i);
        write.dstBinding = offset.binding;
        write.dstSet = descriptorSet;
        write.pImageInfo = &imageInfo;

        writeDescriptor(context, write);
    }
}

bool ShaderObjectImpl::shouldAllocateConstantBuffer(TransientResourceHeapImpl* transientHeap)
{
    return m_isConstantBufferDirty || m_constantBufferTransientHeap != transientHeap ||
           m_constantBufferTransientHeapVersion != transientHeap->getVersion();
}

Result ShaderObjectImpl::_ensureOrdinaryDataBufferCreatedIfNeeded(
    PipelineCommandEncoder* encoder,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // If data has been changed since last allocation/filling of constant buffer,
    // we will need to allocate a new one.
    //
    if (!shouldAllocateConstantBuffer(encoder->m_commandBuffer->m_transientHeap))
    {
        return SLANG_OK;
    }
    m_isConstantBufferDirty = false;
    m_constantBufferTransientHeap = encoder->m_commandBuffer->m_transientHeap;
    m_constantBufferTransientHeapVersion = encoder->m_commandBuffer->m_transientHeap->getVersion();

    m_constantBufferSize = specializedLayout->getTotalOrdinaryDataSize();
    if (m_constantBufferSize == 0)
    {
        return SLANG_OK;
    }

    // Once we have computed how large the buffer should be, we can allocate
    // it from the transient resource heap.
    //
    SLANG_RETURN_ON_FAIL(encoder->m_commandBuffer->m_transientHeap->allocateConstantBuffer(
        m_constantBufferSize,
        m_constantBuffer,
        m_constantBufferOffset));

    // Once the buffer is allocated, we can use `_writeOrdinaryData` to fill it in.
    //
    // Note that `_writeOrdinaryData` is potentially recursive in the case
    // where this object contains interface/existential-type fields, so we
    // don't need or want to inline it into this call site.
    //
    SLANG_RETURN_ON_FAIL(_writeOrdinaryData(
        encoder,
        m_constantBuffer,
        m_constantBufferOffset,
        m_constantBufferSize,
        specializedLayout));

    return SLANG_OK;
}

Result ShaderObjectImpl::bindAsValue(
    PipelineCommandEncoder* encoder,
    RootBindingContext& context,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // We start by iterating over the "simple" (non-sub-object) binding
    // ranges and writing them to the descriptor sets that are being
    // passed down.
    //
    for (auto bindingRangeInfo : specializedLayout->getBindingRanges())
    {
        BindingOffset rangeOffset = offset;

        auto baseIndex = bindingRangeInfo.baseIndex;
        auto count = (uint32_t)bindingRangeInfo.count;
        switch (bindingRangeInfo.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ExistentialValue:
            break;

        case slang::BindingType::Texture:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writeTextureDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                m_resourceViews.getArrayView(baseIndex, count));
            break;
        case slang::BindingType::MutableTexture:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writeTextureDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                m_resourceViews.getArrayView(baseIndex, count));
            break;
        case slang::BindingType::CombinedTextureSampler:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writeTextureSamplerDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                m_combinedTextureSamplers.getArrayView(baseIndex, count));
            break;

        case slang::BindingType::Sampler:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writeSamplerDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_SAMPLER,
                m_samplers.getArrayView(baseIndex, count));
            break;

        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writePlainBufferDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                m_resourceViews.getArrayView(baseIndex, count));
            break;

        case slang::BindingType::TypedBuffer:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writeTexelBufferDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                m_resourceViews.getArrayView(baseIndex, count));
            break;
        case slang::BindingType::MutableTypedBuffer:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writeTexelBufferDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                m_resourceViews.getArrayView(baseIndex, count));
            break;
        case slang::BindingType::RayTracingAccelerationStructure:
            rangeOffset.bindingSet += bindingRangeInfo.setOffset;
            rangeOffset.binding += bindingRangeInfo.bindingOffset;
            writeAccelerationStructureDescriptor(
                context,
                rangeOffset,
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                m_resourceViews.getArrayView(baseIndex, count));
            break;
        case slang::BindingType::VaryingInput:
        case slang::BindingType::VaryingOutput:
            break;

        default:
            SLANG_ASSERT(!"unsupported binding type");
            return SLANG_FAIL;
            break;
        }
    }

    // Once we've handled the simple binding ranges, we move on to the
    // sub-object ranges, which are generally more involved.
    //
    for (auto const& subObjectRange : specializedLayout->getSubObjectRanges())
    {
        auto const& bindingRangeInfo =
            specializedLayout->getBindingRange(subObjectRange.bindingRangeIndex);
        auto count = bindingRangeInfo.count;
        auto subObjectIndex = bindingRangeInfo.subObjectIndex;

        auto subObjectLayout = subObjectRange.layout;

        // The starting offset to use for the sub-object
        // has already been computed and stored as part
        // of the layout, so we can get to the starting
        // offset for the range easily.
        //
        BindingOffset rangeOffset = offset;
        rangeOffset += subObjectRange.offset;

        BindingOffset rangeStride = subObjectRange.stride;

        switch (bindingRangeInfo.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
            {
                BindingOffset objOffset = rangeOffset;
                for (Index i = 0; i < count; ++i)
                {
                    // Binding a constant buffer sub-object is simple enough:
                    // we just call `bindAsConstantBuffer` on it to bind
                    // the ordinary data buffer (if needed) and any other
                    // bindings it recursively contains.
                    //
                    ShaderObjectImpl* subObject = m_objects[subObjectIndex + i];
                    subObject->bindAsConstantBuffer(encoder, context, objOffset, subObjectLayout);

                    // When dealing with arrays of sub-objects, we need to make
                    // sure to increment the offset for each subsequent object
                    // by the appropriate stride.
                    //
                    objOffset += rangeStride;
                }
            }
            break;
        case slang::BindingType::ParameterBlock:
            {
                BindingOffset objOffset = rangeOffset;
                for (Index i = 0; i < count; ++i)
                {
                    // The case for `ParameterBlock<X>` is not that different
                    // from `ConstantBuffer<X>`, except that we call `bindAsParameterBlock`
                    // instead (understandably).
                    //
                    ShaderObjectImpl* subObject = m_objects[subObjectIndex + i];
                    subObject->bindAsParameterBlock(encoder, context, objOffset, subObjectLayout);
                }
            }
            break;

        case slang::BindingType::ExistentialValue:
            // Interface/existential-type sub-object ranges are the most complicated case.
            //
            // First, we can only bind things if we have static specialization information
            // to work with, which is exactly the case where `subObjectLayout` will be
            // non-null.
            //
            if (subObjectLayout)
            {
                // Second, the offset where we want to start binding for existential-type
                // ranges is a bit different, because we don't wnat to bind at the "primary"
                // offset that got passed down, but instead at the "pending" offset.
                //
                // For the purposes of nested binding, what used to be the pending offset
                // will now be used as the primary offset.
                //
                SimpleBindingOffset objOffset = rangeOffset.pending;
                SimpleBindingOffset objStride = rangeStride.pending;
                for (Index i = 0; i < count; ++i)
                {
                    // An existential-type sub-object is always bound just as a value,
                    // which handles its nested bindings and descriptor sets, but
                    // does not deal with ordianry data. The ordinary data should
                    // have been handled as part of the buffer for a parent object
                    // already.
                    //
                    ShaderObjectImpl* subObject = m_objects[subObjectIndex + i];
                    subObject
                        ->bindAsValue(encoder, context, BindingOffset(objOffset), subObjectLayout);
                    objOffset += objStride;
                }
            }
            break;
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
            // No action needed for sub-objects bound though a `StructuredBuffer`.
            break;
        default:
            SLANG_ASSERT(!"unsupported sub-object type");
            return SLANG_FAIL;
            break;
        }
    }

    return SLANG_OK;
}

Result ShaderObjectImpl::allocateDescriptorSets(
    PipelineCommandEncoder* encoder,
    RootBindingContext& context,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    assert(specializedLayout->getOwnDescriptorSets().getCount() <= 1);
    // The number of sets to allocate and their layouts was already pre-computed
    // as part of the shader object layout, so we use that information here.
    //
    for (auto descriptorSetInfo : specializedLayout->getOwnDescriptorSets())
    {
        auto descriptorSetHandle =
            context.descriptorSetAllocator->allocate(descriptorSetInfo.descriptorSetLayout).handle;

        // For each set, we need to write it into the set of descriptor sets
        // being used for binding. This is done both so that other steps
        // in binding can find the set to fill it in, but also so that
        // we can bind all the descriptor sets to the pipeline when the
        // time comes.
        //
        (*context.descriptorSets).add(descriptorSetHandle);
    }

    return SLANG_OK;
}

Result ShaderObjectImpl::bindAsParameterBlock(
    PipelineCommandEncoder* encoder,
    RootBindingContext& context,
    BindingOffset const& inOffset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // Because we are binding into a nested parameter block,
    // any texture/buffer/sampler bindings will now want to
    // write into the sets we allocate for this object and
    // not the sets for any parent object(s).
    //
    BindingOffset offset = inOffset;
    offset.bindingSet = (uint32_t)context.descriptorSets->getCount();
    offset.binding = 0;

    // TODO: We should also be writing to `offset.pending` here,
    // because any resource/sampler bindings related to "pending"
    // data should *also* be writing into the chosen set.
    //
    // The challenge here is that we need to compute the right
    // value for `offset.pending.binding`, so that it writes after
    // all the other bindings.

    // Writing the bindings for a parameter block is relatively easy:
    // we just need to allocate the descriptor set(s) needed for this
    // object and then fill it in like a `ConstantBuffer<X>`.
    //
    SLANG_RETURN_ON_FAIL(allocateDescriptorSets(encoder, context, offset, specializedLayout));

    assert(offset.bindingSet < (uint32_t)context.descriptorSets->getCount());
    SLANG_RETURN_ON_FAIL(bindAsConstantBuffer(encoder, context, offset, specializedLayout));

    return SLANG_OK;
}

Result ShaderObjectImpl::bindOrdinaryDataBufferIfNeeded(
    PipelineCommandEncoder* encoder,
    RootBindingContext& context,
    BindingOffset& ioOffset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // We start by ensuring that the buffer is created, if it is needed.
    //
    SLANG_RETURN_ON_FAIL(_ensureOrdinaryDataBufferCreatedIfNeeded(encoder, specializedLayout));

    // If we did indeed need/create a buffer, then we must bind it into
    // the given `descriptorSet` and update the base range index for
    // subsequent binding operations to account for it.
    //
    if (m_constantBuffer && m_constantBufferSize > 0)
    {
        auto bufferImpl = static_cast<BufferResourceImpl*>(m_constantBuffer);
        writeBufferDescriptor(
            context,
            ioOffset,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            bufferImpl,
            m_constantBufferOffset,
            m_constantBufferSize);
        ioOffset.binding++;
    }

    return SLANG_OK;
}

Result ShaderObjectImpl::bindAsConstantBuffer(
    PipelineCommandEncoder* encoder,
    RootBindingContext& context,
    BindingOffset const& inOffset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // To bind an object as a constant buffer, we first
    // need to bind its ordinary data (if any) into an
    // ordinary data buffer, and then bind it as a "value"
    // which handles any of its recursively-contained bindings.
    //
    // The one detail is taht when binding the ordinary data
    // buffer we need to adjust the `binding` index used for
    // subsequent operations based on whether or not an ordinary
    // data buffer was used (and thus consumed a `binding`).
    //
    BindingOffset offset = inOffset;
    SLANG_RETURN_ON_FAIL(
        bindOrdinaryDataBufferIfNeeded(encoder, context, /*inout*/ offset, specializedLayout));
    SLANG_RETURN_ON_FAIL(bindAsValue(encoder, context, offset, specializedLayout));
    return SLANG_OK;
}

Result ShaderObjectImpl::_getSpecializedLayout(ShaderObjectLayoutImpl** outLayout)
{
    if (!m_specializedLayout)
    {
        SLANG_RETURN_ON_FAIL(_createSpecializedLayout(m_specializedLayout.writeRef()));
    }
    returnRefPtr(outLayout, m_specializedLayout);
    return SLANG_OK;
}

Result ShaderObjectImpl::_createSpecializedLayout(ShaderObjectLayoutImpl** outLayout)
{
    ExtendedShaderObjectType extendedType;
    SLANG_RETURN_ON_FAIL(getSpecializedShaderObjectType(&extendedType));

    auto device = getDevice();
    RefPtr<ShaderObjectLayoutImpl> layout;
    SLANG_RETURN_ON_FAIL(device->getShaderObjectLayout(
        m_layout->m_slangSession,
        extendedType.slangType,
        m_layout->getContainerType(),
        (ShaderObjectLayoutBase**)layout.writeRef()));

    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

Result EntryPointShaderObject::create(
    IDevice* device,
    EntryPointLayout* layout,
    EntryPointShaderObject** outShaderObject)
{
    RefPtr<EntryPointShaderObject> object = new EntryPointShaderObject();
    SLANG_RETURN_ON_FAIL(object->init(device, layout));

    returnRefPtrMove(outShaderObject, object);
    return SLANG_OK;
}

EntryPointLayout* EntryPointShaderObject::getLayout()
{
    return static_cast<EntryPointLayout*>(m_layout.Ptr());
}

Result EntryPointShaderObject::bindAsEntryPoint(
    PipelineCommandEncoder* encoder,
    RootBindingContext& context,
    BindingOffset const& inOffset,
    EntryPointLayout* layout)
{
    BindingOffset offset = inOffset;

    // Any ordinary data in an entry point is assumed to be allocated
    // as a push-constant range.
    //
    // TODO: Can we make this operation not bake in that assumption?
    //
    // TODO: Can/should this function be renamed as just `bindAsPushConstantBuffer`?
    //
    if (m_data.getCount())
    {
        // The index of the push constant range to bind should be
        // passed down as part of the `offset`, and we will increment
        // it here so that any further recursively-contained push-constant
        // ranges use the next index.
        //
        auto pushConstantRangeIndex = offset.pushConstantRange++;

        // Information about the push constant ranges (including offsets
        // and stage flags) was pre-computed for the entire program and
        // stored on the binding context.
        //
        auto const& pushConstantRange = context.pushConstantRanges[pushConstantRangeIndex];

        // We expect that the size of the range as reflected matches the
        // amount of ordinary data stored on this object.
        //
        // TODO: This would not be the case if specialization for interface-type
        // parameters led to the entry point having "pending" ordinary data.
        //
        SLANG_ASSERT(pushConstantRange.size == (uint32_t)m_data.getCount());

        auto pushConstantData = m_data.getBuffer();

        encoder->m_api->vkCmdPushConstants(
            encoder->m_commandBuffer->m_commandBuffer,
            context.pipelineLayout,
            pushConstantRange.stageFlags,
            pushConstantRange.offset,
            pushConstantRange.size,
            pushConstantData);
    }

    // Any remaining bindings in the object can be handled through the
    // "value" case.
    //
    SLANG_RETURN_ON_FAIL(bindAsValue(encoder, context, offset, layout));
    return SLANG_OK;
}

Result EntryPointShaderObject::init(IDevice* device, EntryPointLayout* layout)
{
    SLANG_RETURN_ON_FAIL(Super::init(device, layout));
    return SLANG_OK;
}

RootShaderObjectLayout* RootShaderObjectImpl::getLayout()
{
    return static_cast<RootShaderObjectLayout*>(m_layout.Ptr());
}

RootShaderObjectLayout* RootShaderObjectImpl::getSpecializedLayout()
{
    RefPtr<ShaderObjectLayoutImpl> specializedLayout;
    _getSpecializedLayout(specializedLayout.writeRef());
    return static_cast<RootShaderObjectLayout*>(m_specializedLayout.Ptr());
}

List<RefPtr<EntryPointShaderObject>> const& RootShaderObjectImpl::getEntryPoints() const
{
    return m_entryPoints;
}

GfxCount RootShaderObjectImpl::getEntryPointCount()
{
    return (GfxCount)m_entryPoints.getCount();
}

Result RootShaderObjectImpl::getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint)
{
    returnComPtr(outEntryPoint, m_entryPoints[index]);
    return SLANG_OK;
}

Result RootShaderObjectImpl::copyFrom(IShaderObject* object, ITransientResourceHeap* transientHeap)
{
    SLANG_RETURN_ON_FAIL(Super::copyFrom(object, transientHeap));
    if (auto srcObj = dynamic_cast<MutableRootShaderObject*>(object))
    {
        for (Index i = 0; i < srcObj->m_entryPoints.getCount(); i++)
        {
            m_entryPoints[i]->copyFrom(srcObj->m_entryPoints[i], transientHeap);
        }
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

Result RootShaderObjectImpl::bindAsRoot(
    PipelineCommandEncoder* encoder,
    RootBindingContext& context,
    RootShaderObjectLayout* layout)
{
    BindingOffset offset = {};
    offset.pending = layout->getPendingDataOffset();

    // Note: the operations here are quite similar to what `bindAsParameterBlock` does.
    // The key difference in practice is that we do *not* make use of the adjustment
    // that `bindOrdinaryDataBufferIfNeeded` applied to the offset passed into it.
    //
    // The reason for this difference in behavior is that the layout information
    // for root shader parameters is in practice *already* offset appropriately
    // (so that it ends up using absolute offsets).
    //
    // TODO: One more wrinkle here is that the `ordinaryDataBufferOffset` below
    // might not be correct if `binding=0,set=0` was already claimed via explicit
    // binding information. We should really be getting the offset information for
    // the ordinary data buffer directly from the reflection information for
    // the global scope.

    SLANG_RETURN_ON_FAIL(allocateDescriptorSets(encoder, context, offset, layout));

    BindingOffset ordinaryDataBufferOffset = offset;
    SLANG_RETURN_ON_FAIL(
        bindOrdinaryDataBufferIfNeeded(encoder, context, ordinaryDataBufferOffset, layout));

    SLANG_RETURN_ON_FAIL(bindAsValue(encoder, context, offset, layout));

    auto entryPointCount = layout->getEntryPoints().getCount();
    for (Index i = 0; i < entryPointCount; ++i)
    {
        auto entryPoint = m_entryPoints[i];
        auto const& entryPointInfo = layout->getEntryPoint(i);

        // Note: we do *not* need to add the entry point offset
        // information to the global `offset` because the
        // `RootShaderObjectLayout` has already baked any offsets
        // from the global layout into the `entryPointInfo`.

        entryPoint
            ->bindAsEntryPoint(encoder, context, entryPointInfo.offset, entryPointInfo.layout);
    }

    return SLANG_OK;
}

Result RootShaderObjectImpl::collectSpecializationArgs(ExtendedShaderObjectTypeList& args)
{
    SLANG_RETURN_ON_FAIL(ShaderObjectImpl::collectSpecializationArgs(args));
    for (auto& entryPoint : m_entryPoints)
    {
        SLANG_RETURN_ON_FAIL(entryPoint->collectSpecializationArgs(args));
    }
    return SLANG_OK;
}

Result RootShaderObjectImpl::init(IDevice* device, RootShaderObjectLayout* layout)
{
    SLANG_RETURN_ON_FAIL(Super::init(device, layout));
    m_specializedLayout = nullptr;
    m_entryPoints.clear();
    for (auto entryPointInfo : layout->getEntryPoints())
    {
        RefPtr<EntryPointShaderObject> entryPoint;
        SLANG_RETURN_ON_FAIL(
            EntryPointShaderObject::create(device, entryPointInfo.layout, entryPoint.writeRef()));
        m_entryPoints.add(entryPoint);
    }

    return SLANG_OK;
}

Result RootShaderObjectImpl::_createSpecializedLayout(ShaderObjectLayoutImpl** outLayout)
{
    ExtendedShaderObjectTypeList specializationArgs;
    SLANG_RETURN_ON_FAIL(collectSpecializationArgs(specializationArgs));

    // Note: There is an important policy decision being made here that we need
    // to approach carefully.
    //
    // We are doing two different things that affect the layout of a program:
    //
    // 1. We are *composing* one or more pieces of code (notably the shared global/module
    //    stuff and the per-entry-point stuff).
    //
    // 2. We are *specializing* code that includes generic/existential parameters
    //    to concrete types/values.
    //
    // We need to decide the relative *order* of these two steps, because of how it impacts
    // layout. The layout for `specialize(compose(A,B), X, Y)` is potentially different
    // form that of `compose(specialize(A,X), speciealize(B,Y))`, even when both are
    // semantically equivalent programs.
    //
    // Right now we are using the first option: we are first generating a full composition
    // of all the code we plan to use (global scope plus all entry points), and then
    // specializing it to the concatenated specialization argumenst for all of that.
    //
    // In some cases, though, this model isn't appropriate. For example, when dealing with
    // ray-tracing shaders and local root signatures, we really want the parameters of each
    // entry point (actually, each entry-point *group*) to be allocated distinct storage,
    // which really means we want to compute something like:
    //
    //      SpecializedGlobals = specialize(compose(ModuleA, ModuleB, ...), X, Y, ...)
    //
    //      SpecializedEP1 = compose(SpecializedGlobals, specialize(EntryPoint1, T, U, ...))
    //      SpecializedEP2 = compose(SpecializedGlobals, specialize(EntryPoint2, A, B, ...))
    //
    // Note how in this case all entry points agree on the layout for the shared/common
    // parmaeters, but their layouts are also independent of one another.
    //
    // Furthermore, in this example, loading another entry point into the system would not
    // rquire re-computing the layouts (or generated kernel code) for any of the entry
    // points that had already been loaded (in contrast to a compose-then-specialize
    // approach).
    //
    ComPtr<slang::IComponentType> specializedComponentType;
    ComPtr<slang::IBlob> diagnosticBlob;
    auto result = getLayout()->getSlangProgram()->specialize(
        specializationArgs.components.getArrayView().getBuffer(),
        specializationArgs.getCount(),
        specializedComponentType.writeRef(),
        diagnosticBlob.writeRef());

    // TODO: print diagnostic message via debug output interface.

    if (result != SLANG_OK)
        return result;

    auto slangSpecializedLayout = specializedComponentType->getLayout();
    RefPtr<RootShaderObjectLayout> specializedLayout;
    RootShaderObjectLayout::create(
        static_cast<DeviceImpl*>(getRenderer()),
        specializedComponentType,
        slangSpecializedLayout,
        specializedLayout.writeRef());

    // Note: Computing the layout for the specialized program will have also computed
    // the layouts for the entry points, and we really need to attach that information
    // to them so that they don't go and try to compute their own specializations.
    //
    // TODO: Well, if we move to the specialization model described above then maybe
    // we *will* want entry points to do their own specialization work...
    //
    auto entryPointCount = m_entryPoints.getCount();
    for (Index i = 0; i < entryPointCount; ++i)
    {
        auto entryPointInfo = specializedLayout->getEntryPoint(i);
        auto entryPointVars = m_entryPoints[i];

        entryPointVars->m_specializedLayout = entryPointInfo.layout;
    }

    returnRefPtrMove(outLayout, specializedLayout);
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
