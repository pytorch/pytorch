// metal-shader-object.cpp
#include "metal-shader-object.h"

#include "metal-device.h"
#include "metal-sampler.h"

namespace gfx
{

using namespace Slang;

namespace metal
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

ShaderObjectImpl::~ShaderObjectImpl() {}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::setData(ShaderOffset const& inOffset, void const* data, size_t inSize)
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
    m_isArgumentBufferDirty = true;
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::setResource(ShaderOffset const& offset, IResourceView* resourceView)
{
    if (offset.bindingRangeIndex < 0)
        return SLANG_E_INVALID_ARG;
    auto layout = getLayout();
    if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
        return SLANG_E_INVALID_ARG;
    auto& bindingRange = layout->getBindingRange(offset.bindingRangeIndex);

    auto resourceViewImpl = static_cast<ResourceViewImpl*>(resourceView);
    switch (bindingRange.bindingType)
    {
    case slang::BindingType::Texture:
    case slang::BindingType::MutableTexture:
        SLANG_ASSERT(resourceViewImpl->m_type == ResourceViewImpl::ViewType::Texture);
        m_textures[bindingRange.baseIndex + offset.bindingArrayIndex] =
            static_cast<TextureResourceViewImpl*>(resourceView);

        // For parameter blocks, we just need to set the resource ID of the texture to argument
        // buffer
        if (getLayout()->isParameterBlock())
        {
            auto resourceId =
                static_cast<TextureResourceViewImpl*>(resourceView)->m_textureView->gpuResourceID();
            setData(offset, &resourceId, sizeof(resourceId));
        }
        break;
    case slang::BindingType::RawBuffer:
    case slang::BindingType::ConstantBuffer:
    case slang::BindingType::MutableRawBuffer:
        SLANG_ASSERT(resourceViewImpl->m_type == ResourceViewImpl::ViewType::Buffer);
        m_buffers[bindingRange.baseIndex + offset.bindingArrayIndex] =
            static_cast<BufferResourceViewImpl*>(resourceView);

        // For parameter blocks, we just need to set the GPU address of the buffer to argument
        // buffer
        if (getLayout()->isParameterBlock())
        {
            DeviceAddress gpuAddress =
                static_cast<BufferResourceViewImpl*>(resourceView)->m_buffer->getDeviceAddress();
            setData(offset, &gpuAddress, sizeof(gpuAddress));
        }
        break;
    case slang::BindingType::TypedBuffer:
    case slang::BindingType::MutableTypedBuffer:
        SLANG_ASSERT(!"Not implemented");
        // SLANG_ASSERT(resourceViewImpl->m_type == ResourceViewImpl::ViewType::TexelBuffer);
        // m_textures[bindingRange.baseIndex + offset.bindingArrayIndex] =
        // static_cast<TextureResourceViewImpl*>(resourceView);
        break;
    }
    m_isArgumentBufferDirty = true;
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::setSampler(ShaderOffset const& offset, ISamplerState* sampler)
{
    if (offset.bindingRangeIndex < 0)
        return SLANG_E_INVALID_ARG;
    auto layout = getLayout();
    if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
        return SLANG_E_INVALID_ARG;
    auto& bindingRange = layout->getBindingRange(offset.bindingRangeIndex);

    m_samplers[bindingRange.baseIndex + offset.bindingArrayIndex] =
        static_cast<SamplerStateImpl*>(sampler);

    // For parameter blocks, we just need to set the GPU address of the buffer to argument buffer
    if (layout->isParameterBlock())
    {
        auto resourceId = static_cast<SamplerStateImpl*>(sampler)->m_samplerState->gpuResourceID();
        setData(offset, &resourceId, sizeof(resourceId));
    }
    m_isArgumentBufferDirty = true;
    return SLANG_OK;
}

Result ShaderObjectImpl::init(IDevice* device, ShaderObjectLayoutImpl* layout)
{
    m_layout = layout;

    // If the layout tells us that there is any uniform data,
    // then we will allocate a CPU memory buffer to hold that data
    // while it is being set from the host.
    //
    // Once the user is done setting the parameters/fields of this
    // shader object, we will produce a GPU-memory version of the
    // uniform data (which includes values from this object and
    // any existential-type sub-objects).
    //
    size_t uniformSize = 0;
    if (layout->isParameterBlock())
        uniformSize = layout->getParameterBlockTypeLayout()->getSize();
    else
        uniformSize = layout->getElementTypeLayout()->getSize();

    if (uniformSize)
    {
        m_data.setCount(uniformSize);
        memset(m_data.getBuffer(), 0, uniformSize);
    }

    m_buffers.setCount(layout->getBufferCount());
    m_textures.setCount(layout->getTextureCount());
    m_samplers.setCount(layout->getSamplerCount());

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

            if (bindingRangeInfo.bindingType == slang::BindingType::ParameterBlock ||
                bindingRangeInfo.bindingType == slang::BindingType::ConstantBuffer)
                subObjectLayout->setIsParameterBlock();

            SLANG_RETURN_ON_FAIL(
                ShaderObjectImpl::create(device, subObjectLayout, subObject.writeRef()));
            m_objects[bindingRangeInfo.subObjectIndex + i] = subObject;
        }
    }
    m_isArgumentBufferDirty = true;
    return SLANG_OK;
}

Result ShaderObjectImpl::_writeOrdinaryData(
    void* dest,
    size_t destSize,
    ShaderObjectLayoutImpl* layout)
{
    // We start by simply writing in the ordinary data contained directly in this object.
    //
    auto src = m_data.getBuffer();
    auto srcSize = size_t(m_data.getCount());
    SLANG_ASSERT(srcSize <= destSize);
    memcpy(dest, src, srcSize);

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
    for (auto const& subObjectRangeInfo : layout->getSubObjectRanges())
    {
        Index subObjectRangeIndex = subObjectRangeCounter++;
        auto const& bindingRangeInfo =
            layout->getBindingRange(subObjectRangeInfo.bindingRangeIndex);

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
        // offset and stride for the location of the pending data allocation in the specialized
        // type layout, which will store the values for this sub-object range.
        //
        // TODO: The reflection API functions we are assuming here haven't been implemented
        // yet, so the functions being called here are stubs.
        //
        // TODO: It might not be that a single sub-object range can reliably map to a single
        // contiguous array with a single stride; we need to carefully consider what the layout
        // logic does for complex cases with multiple layers of nested arrays and structures.
        //
        size_t subObjectRangePendingDataOffset = subObjectRangeInfo.offset.pendingOrdinaryData;
        size_t subObjectRangePendingDataStride = subObjectRangeInfo.stride.pendingOrdinaryData;

        // If the range doesn't actually need/use the "pending" allocation at all, then
        // we need to detect that case and skip such ranges.
        //
        // TODO: This should probably be handled on a per-object basis by caching a "does it fit?"
        // bit as part of the information for bound sub-objects, given that we already
        // compute the "does it fit?" status as part of `setObject()`.
        //
        if (subObjectRangePendingDataOffset == 0)
            continue;

        for (Slang::Index i = 0; i < count; ++i)
        {
            auto subObject = m_objects[bindingRangeInfo.subObjectIndex + i];

            ShaderObjectLayoutImpl* subObjectLayout = subObject->getLayout();

            auto subObjectOffset =
                subObjectRangePendingDataOffset + i * subObjectRangePendingDataStride;

            auto subObjectDest = (char*)dest + subObjectOffset;

            subObject->_writeOrdinaryData(
                subObjectDest,
                destSize - subObjectOffset,
                subObjectLayout);
        }
    }
    return SLANG_OK;
}

Result ShaderObjectImpl::_ensureOrdinaryDataBufferCreatedIfNeeded(
    DeviceImpl* device,
    ShaderObjectLayoutImpl* layout)
{
    auto ordinaryDataSize = layout->getTotalOrdinaryDataSize();
    if (ordinaryDataSize == 0)
        return SLANG_OK;

    // If we have already created a buffer to hold ordinary data, then we should
    // simply re-use that buffer rather than re-create it.
    if (!m_ordinaryDataBuffer)
    {
        ComPtr<IBufferResource> bufferResourcePtr;
        IBufferResource::Desc bufferDesc = {};
        bufferDesc.type = IResource::Type::Buffer;
        bufferDesc.sizeInBytes = ordinaryDataSize;
        bufferDesc.defaultState = ResourceState::ConstantBuffer;
        bufferDesc.allowedStates =
            ResourceStateSet(ResourceState::ConstantBuffer, ResourceState::CopyDestination);
        bufferDesc.memoryType = MemoryType::Upload;
        SLANG_RETURN_ON_FAIL(
            device->createBufferResource(bufferDesc, nullptr, bufferResourcePtr.writeRef()));
        m_ordinaryDataBuffer = static_cast<BufferResourceImpl*>(bufferResourcePtr.get());
    }

    if (m_isConstantBufferDirty)
    {
        // Once the buffer is allocated, we can use `_writeOrdinaryData` to fill it in.
        //
        // Note that `_writeOrdinaryData` is potentially recursive in the case
        // where this object contains interface/existential-type fields, so we
        // don't need or want to inline it into this call site.
        //

        MemoryRange range = {0, ordinaryDataSize};
        void* ordinaryData;
        SLANG_RETURN_ON_FAIL(m_ordinaryDataBuffer->map(&range, &ordinaryData));
        auto result = _writeOrdinaryData(ordinaryData, ordinaryDataSize, layout);
        m_ordinaryDataBuffer->unmap(&range);
        m_isConstantBufferDirty = false;
        return result;
    }
    return SLANG_OK;
}

Result ShaderObjectImpl::_bindOrdinaryDataBufferIfNeeded(
    BindingContext* context,
    BindingOffset& ioOffset,
    ShaderObjectLayoutImpl* layout)
{
    // We start by ensuring that the buffer is created, if it is needed.
    //
    SLANG_RETURN_ON_FAIL(_ensureOrdinaryDataBufferCreatedIfNeeded(context->device, layout));

    // If we did indeed need/create a buffer, then we must bind it
    // into root binding state.
    //
    if (m_ordinaryDataBuffer)
    {
        context->setBuffer(m_ordinaryDataBuffer->m_buffer.get(), ioOffset.buffer);
        ioOffset.buffer++;
    }

    return SLANG_OK;
}

void ShaderObjectImpl::writeOrdinaryDataIntoArgumentBuffer(
    slang::TypeLayoutReflection* argumentBufferTypeLayout,
    slang::TypeLayoutReflection* defaultTypeLayout,
    uint8_t* argumentBuffer,
    uint8_t* srcData)
{
    // If we are pure data, just copy it over from srcData.
    if (defaultTypeLayout->getCategoryCount() == 1)
    {
        switch (defaultTypeLayout->getCategoryByIndex(0))
        {
        case slang::ParameterCategory::Uniform:
            // Just copy the uniform data
            memcpy(argumentBuffer, srcData, defaultTypeLayout->getSize());
            break;
        }
        return;
    }

    for (unsigned int i = 0; i < argumentBufferTypeLayout->getFieldCount(); i++)
    {
        auto argumentBufferField = argumentBufferTypeLayout->getFieldByIndex(i);
        auto defaultLayoutField = defaultTypeLayout->getFieldByIndex(i);
        // If the field is mixed type, recurse.
        writeOrdinaryDataIntoArgumentBuffer(
            argumentBufferField->getTypeLayout(),
            defaultLayoutField->getTypeLayout(),
            argumentBuffer + argumentBufferField->getOffset(),
            srcData + defaultLayoutField->getOffset());
    }
}

BufferResourceImpl* ShaderObjectImpl::_ensureArgumentBufferUpToDate(
    BindingContext* context,
    DeviceImpl* device,
    ShaderObjectLayoutImpl* layout)
{
    auto typeLayout = layout->getParameterBlockTypeLayout();

    // If we have already created a buffer to hold the parmaeter block, then we should
    // simply re-use that buffer rather than re-create it.
    if (!m_argumentBuffer)
    {
        ComPtr<IBufferResource> bufferResourcePtr;
        IBufferResource::Desc bufferDesc = {};
        bufferDesc.type = IResource::Type::Buffer;
        bufferDesc.sizeInBytes = typeLayout->getSize();
        bufferDesc.defaultState = ResourceState::ConstantBuffer;
        bufferDesc.allowedStates =
            ResourceStateSet(ResourceState::ConstantBuffer, ResourceState::CopyDestination);
        bufferDesc.memoryType = MemoryType::Upload;
        SLANG_RETURN_NULL_ON_FAIL(
            device->createBufferResource(bufferDesc, nullptr, bufferResourcePtr.writeRef()));
        m_argumentBuffer = static_cast<BufferResourceImpl*>(bufferResourcePtr.get());
    }

    if (m_isArgumentBufferDirty)
    {
        // Once the buffer is allocated, we can fill it in with the uniform data
        // and resource bindings we have tracked, using `typeLayout` to obtain
        // the offsets for each field.
        //
        auto dataSize = typeLayout->getSize();
        MemoryRange range = {0, dataSize};
        void* argumentData;
        SLANG_RETURN_NULL_ON_FAIL(m_argumentBuffer->map(&range, &argumentData));

        // For parameter blocks, all the fields are flattened as ordinary data, so the size of the
        // m_data must be equal to the size of the argument buffer, we just need to copy the data
        // from m_data to argumentData, the only thing we need to specially handle is the parameter
        // block and constant buffer, which will be a represented as device pointer in the argument
        // buffer, we have to set the address of the argument buffer of nested parameter block to
        // the corresponding offset in the argument buffer
        SLANG_ASSERT(m_data.getCount() == dataSize);
        memcpy(argumentData, m_data.getBuffer(), dataSize);

        // Special handle the parameter block and constant buffer
        for (uint32_t i = 0; i < typeLayout->getFieldCount(); i++)
        {
            auto field = typeLayout->getFieldByIndex(i);
            auto kind = field->getTypeLayout()->getKind();
            switch (kind)
            {
            case slang::TypeReflection::Kind::ConstantBuffer:
            case slang::TypeReflection::Kind::ParameterBlock:
                {
                    // set address of argument buffer of nested parameter block to corresponding
                    // offset in argument buffer
                    auto offset = field->getOffset();
                    uint32_t bindingRangeIndex = typeLayout->getFieldBindingRangeOffset(i);
                    auto bindingRange = layout->getBindingRange(bindingRangeIndex);
                    auto subObjectIndex = bindingRange.subObjectIndex;
                    auto subObject = m_objects[subObjectIndex];
                    BufferResourceImpl* argumentBufferPtr =
                        subObject->_ensureArgumentBufferUpToDate(
                            context,
                            device,
                            subObject->getLayout());
                    if (argumentBufferPtr)
                    {
                        uint8_t* argumentBuffer = (uint8_t*)argumentData + offset;
                        gfx::DeviceAddress bufferAddr = argumentBufferPtr->getDeviceAddress();
                        memcpy(argumentBuffer, &bufferAddr, sizeof(bufferAddr));

                        MTL::Resource const* resource[] = {argumentBufferPtr->m_buffer.get()};
                        // Nested parameter block and constant buffer is also bindless resource, we
                        // need to inform Metal to hazard track the resource
                        context->useResources(
                            resource,
                            1,
                            MTL::ResourceUsageWrite | MTL::ResourceUsageRead);
                    }
                    break;
                }
            default:
                break;
            }
        }

        // Handle bindless resources
        List<MTL::Resource const*> resources;
        for (uint32_t i = 0; i < m_buffers.getCount(); i++)
        {
            if (m_buffers[i])
            {
                MTL::Buffer* mtlBuffer = m_buffers[i]->m_buffer->m_buffer.get();
                resources.add(mtlBuffer);
            }
        }

        for (uint32_t i = 0; i < m_textures.getCount(); i++)
        {
            if (m_textures[i])
            {
                MTL::Texture* mtlTexture = m_textures[i]->m_texture->m_texture.get();
                resources.add(mtlTexture);
            }
        }
        // It's important to call useResources because Metal will not automatically do the hazard
        // tracking for bindless resources, we have to call useResources to inform Metal to track
        // the resources.
        context->useResources(
            resources.getBuffer(),
            resources.getCount(),
            MTL::ResourceUsageWrite | MTL::ResourceUsageRead);

        m_argumentBuffer->unmap(&range);
        m_isArgumentBufferDirty = false;
    }

    return m_argumentBuffer.get();
}

Result ShaderObjectImpl::bindAsParameterBlock(
    BindingContext* context,
    BindingOffset const& inOffset,
    ShaderObjectLayoutImpl* layout)
{
    if (!context->device->m_hasArgumentBufferTier2)
        return SLANG_FAIL;

    auto argumentBuffer = _ensureArgumentBufferUpToDate(context, context->device, layout);

    if (m_argumentBuffer)
    {
        context->setBuffer(m_argumentBuffer->m_buffer.get(), inOffset.buffer);
    }
    return SLANG_OK;
}

Result ShaderObjectImpl::bindAsConstantBuffer(
    BindingContext* context,
    BindingOffset const& inOffset,
    ShaderObjectLayoutImpl* layout)
{
    // When binding a `ConstantBuffer<X>` we need to first bind a constant
    // buffer for any "ordinary" data in `X`, and then bind the remaining
    // resources and sub-objects.
    //
    BindingOffset offset = inOffset;
    SLANG_RETURN_ON_FAIL(_bindOrdinaryDataBufferIfNeeded(context, /*inout*/ offset, layout));

    // Once the ordinary data buffer is bound, we can move on to binding
    // the rest of the state, which can use logic shared with the case
    // for interface-type sub-object ranges.
    //
    // Note that this call will use the `inOffset` value instead of the offset
    // modified by `_bindOrindaryDataBufferIfNeeded', because the indexOffset in
    // the binding range should already take care of the offset due to the default
    // cbuffer.
    //
    SLANG_RETURN_ON_FAIL(bindAsValue(context, inOffset, layout));

    return SLANG_OK;
}

Result ShaderObjectImpl::bindAsValue(
    BindingContext* context,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* layout)
{
    // We start by iterating over the binding ranges in this type, isolating
    // just those ranges that represent buffers, textures, and samplers.
    // In each loop we will bind the values stored for those binding ranges
    // to the correct metal resource indices (based on the `registerOffset` field
    // stored in the bindinge range).

    for (auto bindingRangeIndex : layout->getBufferRanges())
    {
        auto const& bindingRange = layout->getBindingRange(bindingRangeIndex);
        auto count = (uint32_t)bindingRange.count;
        auto baseIndex = (uint32_t)bindingRange.baseIndex;
        auto registerOffset = bindingRange.registerOffset + offset.buffer;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto buffer = m_buffers[baseIndex + i];
            context->setBuffer(
                buffer ? buffer->m_buffer->m_buffer.get() : nullptr,
                registerOffset + i);
        }
    }

    for (auto bindingRangeIndex : layout->getTextureRanges())
    {
        auto const& bindingRange = layout->getBindingRange(bindingRangeIndex);
        auto count = (uint32_t)bindingRange.count;
        auto baseIndex = (uint32_t)bindingRange.baseIndex;
        auto registerOffset = bindingRange.registerOffset + offset.texture;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto texture = m_textures[baseIndex + i];
            context->setTexture(
                texture ? texture->m_textureView.get() : nullptr,
                registerOffset + i);
        }
    }

    for (auto bindingRangeIndex : layout->getSamplerRanges())
    {
        auto const& bindingRange = layout->getBindingRange(bindingRangeIndex);
        auto count = (uint32_t)bindingRange.count;
        auto baseIndex = (uint32_t)bindingRange.baseIndex;
        auto registerOffset = bindingRange.registerOffset + offset.sampler;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto sampler = m_samplers[baseIndex + i];
            context->setSampler(
                sampler ? sampler->m_samplerState.get() : nullptr,
                registerOffset + i);
        }
    }

    // Once all the simple binding ranges are dealt with, we will bind
    // all of the sub-objects in sub-object ranges.
    //
    for (auto const& subObjectRange : layout->getSubObjectRanges())
    {
        auto subObjectLayout = subObjectRange.layout;
        auto const& bindingRange = layout->getBindingRange(subObjectRange.bindingRangeIndex);
        Index count = bindingRange.count;
        Index subObjectIndex = bindingRange.subObjectIndex;

        // The starting offset for a sub-object range was computed
        // from Slang reflection information, so we can apply it here.
        //
        BindingOffset rangeOffset = offset;
        rangeOffset += subObjectRange.offset;

        // Similarly, the "stride" between consecutive objects in
        // the range was also pre-computed.
        //
        BindingOffset rangeStride = subObjectRange.stride;

        switch (bindingRange.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
            {
                BindingOffset objOffset = rangeOffset;
                for (Index i = 0; i < count; ++i)
                {
                    auto subObject = m_objects[subObjectIndex + i];

                    // Unsurprisingly, we bind each object in the range as
                    // a constant buffer.
                    //
                    SLANG_RETURN_ON_FAIL(
                        subObject->bindAsConstantBuffer(context, objOffset, subObjectLayout));

                    objOffset += rangeStride;
                }
                break;
            }
        case slang::BindingType::ParameterBlock:
            {
                BindingOffset objOffset = rangeOffset;
                for (Index i = 0; i < count; ++i)
                {
                    auto subObject = m_objects[subObjectIndex + i];
                    SLANG_RETURN_ON_FAIL(
                        subObject->bindAsParameterBlock(context, objOffset, subObjectLayout));
                    objOffset += rangeStride;
                }
            }
            break;

#if 0
        case slang::BindingType::ExistentialValue:
            // We can only bind information for existential-typed sub-object
            // ranges if we have a static type that we are able to specialize to.
            //
            if (subObjectLayout)
            {
                // The data for objects in this range will always be bound into
                // the "pending" allocation for the parent block/buffer/object.
                // As a result, the offset for the first object in the range
                // will come from the `pending` part of the range's offset.
                //
                SimpleBindingOffset objOffset = rangeOffset.pending;
                SimpleBindingOffset objStride = rangeStride.pending;

                for (Index i = 0; i < count; ++i)
                {
                    auto subObject = m_objects[subObjectIndex + i];
                    subObject->bindAsValue(context, BindingOffset(objOffset), subObjectLayout);

                    objOffset += objStride;
                }
            }
            break;
#endif

        default:
            break;
        }
    }

    return SLANG_OK;
}

Result RootShaderObjectImpl::create(
    IDevice* device,
    RootShaderObjectLayoutImpl* layout,
    RootShaderObjectImpl** outShaderObject)
{
    RefPtr<RootShaderObjectImpl> object = new RootShaderObjectImpl();
    SLANG_RETURN_ON_FAIL(object->init(device, layout));

    returnRefPtrMove(outShaderObject, object);
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

Result RootShaderObjectImpl::bindAsRoot(BindingContext* context, RootShaderObjectLayoutImpl* layout)
{
    // When binding an entire root shader object, we need to deal with
    // the way that specialization might have allocated space for "pending"
    // parameter data after all the primary parameters.
    //
    // We start by initializing an offset that will store zeros for the
    // primary data, an the computed offset from the specialized layout
    // for pending data.
    //
    BindingOffset offset;
#if 0
    offset.pending = layout->getPendingDataOffset();
#endif

    // Note: We could *almost* call `bindAsConstantBuffer()` here to bind
    // the state of the root object itself, but there is an important
    // detail that means we can't:
    //
    // The `_bindOrdinaryDataBufferIfNeeded` operation automatically
    // increments the offset parameter if it binds a buffer, so that
    // subsequently bindings will be adjusted. However, the reflection
    // information computed for root shader parameters is absolute rather
    // than relative to the default constant buffer (if any).
    //
    // TODO: Quite technically, the ordinary data buffer for the global
    // scope is *not* guaranteed to be at offset zero, so this logic should
    // really be querying an appropriate absolute offset from `layout`.
    //
#if 0
    BindingOffset ordinaryDataBufferOffset = offset;
    SLANG_RETURN_ON_FAIL(_bindOrdinaryDataBufferIfNeeded(context, /*inout*/ ordinaryDataBufferOffset, layout));
#endif
    SLANG_RETURN_ON_FAIL(bindAsValue(context, offset, layout));

    // Once the state stored in the root shader object itself has been bound,
    // we turn our attention to the entry points and their parameters.
    //
    auto entryPointCount = m_entryPoints.getCount();
    for (Index i = 0; i < entryPointCount; ++i)
    {
        auto entryPoint = m_entryPoints[i];
        auto const& entryPointInfo = layout->getEntryPoint(i);

        // Each entry point will be bound at some offset relative to where
        // the root shader parameters start.
        //
        BindingOffset entryPointOffset = offset;
        entryPointOffset += entryPointInfo.offset;

        // An entry point can simply be bound as a constant buffer, because
        // the absolute offsets as are used for the global scope do not apply
        // (because entry points don't need to deal with explicit bindings).
        //
        SLANG_RETURN_ON_FAIL(
            entryPoint->bindAsConstantBuffer(context, entryPointOffset, entryPointInfo.layout));
    }

    return SLANG_OK;
}

Result RootShaderObjectImpl::init(IDevice* device, RootShaderObjectLayoutImpl* layout)
{
    SLANG_RETURN_ON_FAIL(Super::init(device, layout));
    m_entryPoints.clear();
    for (auto entryPointInfo : layout->getEntryPoints())
    {
        RefPtr<ShaderObjectImpl> entryPoint;
        SLANG_RETURN_ON_FAIL(
            ShaderObjectImpl::create(device, entryPointInfo.layout, entryPoint.writeRef()));
        m_entryPoints.add(entryPoint);
    }

    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
