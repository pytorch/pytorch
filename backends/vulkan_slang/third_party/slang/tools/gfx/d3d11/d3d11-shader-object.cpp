// d3d11-shader-object.cpp
#include "d3d11-shader-object.h"

#include "d3d11-device.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
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
    if (D3DUtil::isUAVBinding(bindingRange.bindingType))
    {
        SLANG_ASSERT(resourceViewImpl->m_type == ResourceViewImpl::Type::UAV);
        m_uavs[bindingRange.baseIndex + offset.bindingArrayIndex] =
            static_cast<UnorderedAccessViewImpl*>(resourceView);
    }
    else
    {
        SLANG_ASSERT(resourceViewImpl->m_type == ResourceViewImpl::Type::SRV);
        m_srvs[bindingRange.baseIndex + offset.bindingArrayIndex] =
            static_cast<ShaderResourceViewImpl*>(resourceView);
    }
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
    size_t uniformSize = layout->getElementTypeLayout()->getSize();
    if (uniformSize)
    {
        m_data.setCount(uniformSize);
        memset(m_data.getBuffer(), 0, uniformSize);
    }

    m_srvs.setCount(layout->getSRVCount());
    m_samplers.setCount(layout->getSamplerCount());
    m_uavs.setCount(layout->getUAVCount());

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
    void* dest,
    size_t destSize,
    ShaderObjectLayoutImpl* specializedLayout)
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

            RefPtr<ShaderObjectLayoutImpl> subObjectLayout;
            SLANG_RETURN_ON_FAIL(subObject->_getSpecializedLayout(subObjectLayout.writeRef()));

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
    ShaderObjectLayoutImpl* specializedLayout)
{
    auto specializedOrdinaryDataSize = specializedLayout->getTotalOrdinaryDataSize();
    if (specializedOrdinaryDataSize == 0)
        return SLANG_OK;

    // If we have already created a buffer to hold ordinary data, then we should
    // simply re-use that buffer rather than re-create it.
    if (!m_ordinaryDataBuffer)
    {
        ComPtr<IBufferResource> bufferResourcePtr;
        IBufferResource::Desc bufferDesc = {};
        bufferDesc.type = IResource::Type::Buffer;
        bufferDesc.sizeInBytes = specializedOrdinaryDataSize;
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

        auto ordinaryData = device->map(m_ordinaryDataBuffer, gfx::MapFlavor::WriteDiscard);
        auto result =
            _writeOrdinaryData(ordinaryData, specializedOrdinaryDataSize, specializedLayout);
        device->unmap(m_ordinaryDataBuffer, 0, specializedOrdinaryDataSize);
        m_isConstantBufferDirty = false;
        return result;
    }
    return SLANG_OK;
}

Result ShaderObjectImpl::_bindOrdinaryDataBufferIfNeeded(
    BindingContext* context,
    BindingOffset& ioOffset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // We start by ensuring that the buffer is created, if it is needed.
    //
    SLANG_RETURN_ON_FAIL(
        _ensureOrdinaryDataBufferCreatedIfNeeded(context->device, specializedLayout));

    // If we did indeed need/create a buffer, then we must bind it
    // into root binding state.
    //
    if (m_ordinaryDataBuffer)
    {
        context->setCBV(ioOffset.cbv, m_ordinaryDataBuffer->m_buffer);
        ioOffset.cbv++;
    }

    return SLANG_OK;
}

Result ShaderObjectImpl::bindAsConstantBuffer(
    BindingContext* context,
    BindingOffset const& inOffset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // When binding a `ConstantBuffer<X>` we need to first bind a constant
    // buffer for any "ordinary" data in `X`, and then bind the remaining
    // resources and sub-objects.
    //
    BindingOffset offset = inOffset;
    SLANG_RETURN_ON_FAIL(
        _bindOrdinaryDataBufferIfNeeded(context, /*inout*/ offset, specializedLayout));

    // Once the ordinary data buffer is bound, we can move on to binding
    // the rest of the state, which can use logic shared with the case
    // for interface-type sub-object ranges.
    //
    // Note that this call will use the `inOffset` value instead of the offset
    // modified by `_bindOrindaryDataBufferIfNeeded', because the indexOffset in
    // the binding range should already take care of the offset due to the default
    // cbuffer.
    //
    SLANG_RETURN_ON_FAIL(bindAsValue(context, inOffset, specializedLayout));

    return SLANG_OK;
}

Result ShaderObjectImpl::bindAsValue(
    BindingContext* context,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // We start by iterating over the binding ranges in this type, isolating
    // just those ranges that represent SRVs, UAVs, and samplers.
    // In each loop we will bind the values stored for those binding ranges
    // to the correct D3D11 register (based on the `registerOffset` field
    // stored in the bindinge range).
    //
    // TODO: These loops could be optimized if we stored parallel arrays
    // for things like `m_srvs` so that we directly store an array of
    // `ID3D11ShaderResourceView*` where each entry matches the `gfx`-level
    // object that was bound (or holds null if nothing is bound).
    // In that case, we could perform a single `setSRVs()` call for each
    // binding range.
    //
    // TODO: More ambitiously, if the Slang layout algorithm could be modified
    // so that non-sub-object binding ranges are guaranteed to be contiguous
    // then a *single* `setSRVs()` call could set all of the SRVs for an object
    // at once.

    for (auto bindingRangeIndex : specializedLayout->getSRVRanges())
    {
        auto const& bindingRange = specializedLayout->getBindingRange(bindingRangeIndex);
        auto count = (uint32_t)bindingRange.count;
        auto baseIndex = (uint32_t)bindingRange.baseIndex;
        auto registerOffset = bindingRange.registerOffset + offset.srv;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto srv = m_srvs[baseIndex + i];
            context->setSRV(registerOffset + i, srv ? srv->m_srv : nullptr);
        }
    }

    for (auto bindingRangeIndex : specializedLayout->getUAVRanges())
    {
        auto const& bindingRange = specializedLayout->getBindingRange(bindingRangeIndex);
        auto count = (uint32_t)bindingRange.count;
        auto baseIndex = (uint32_t)bindingRange.baseIndex;
        auto registerOffset = bindingRange.registerOffset + offset.uav;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto uav = m_uavs[baseIndex + i];
            context->setUAV(registerOffset + i, uav ? uav->m_uav : nullptr);
        }
    }

    for (auto bindingRangeIndex : specializedLayout->getSamplerRanges())
    {
        auto const& bindingRange = specializedLayout->getBindingRange(bindingRangeIndex);
        auto count = (uint32_t)bindingRange.count;
        auto baseIndex = (uint32_t)bindingRange.baseIndex;
        auto registerOffset = bindingRange.registerOffset + offset.sampler;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto sampler = m_samplers[baseIndex + i];
            context->setSampler(registerOffset + i, sampler ? sampler->m_sampler.get() : nullptr);
        }
    }

    // Once all the simple binding ranges are dealt with, we will bind
    // all of the sub-objects in sub-object ranges.
    //
    for (auto const& subObjectRange : specializedLayout->getSubObjectRanges())
    {
        auto subObjectLayout = subObjectRange.layout;
        auto const& bindingRange =
            specializedLayout->getBindingRange(subObjectRange.bindingRangeIndex);
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
            // For D3D11-compatible compilation targets, the Slang compiler
            // treats the `ConstantBuffer<T>` and `ParameterBlock<T>` types the same.
            //
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
            {
                BindingOffset objOffset = rangeOffset;
                for (Index i = 0; i < count; ++i)
                {
                    auto subObject = m_objects[subObjectIndex + i];

                    // Unsurprisingly, we bind each object in the range as
                    // a constant buffer.
                    //
                    subObject->bindAsConstantBuffer(context, objOffset, subObjectLayout);

                    objOffset += rangeStride;
                }
            }
            break;

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

        default:
            break;
        }
    }

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

    auto renderer = getRenderer();
    RefPtr<ShaderObjectLayoutImpl> layout;
    SLANG_RETURN_ON_FAIL(renderer->getShaderObjectLayout(
        m_layout->m_slangSession,
        extendedType.slangType,
        m_layout->getContainerType(),
        (ShaderObjectLayoutBase**)layout.writeRef()));

    returnRefPtrMove(outLayout, layout);
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

Result RootShaderObjectImpl::bindAsRoot(
    BindingContext* context,
    RootShaderObjectLayoutImpl* specializedLayout)
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
    offset.pending = specializedLayout->getPendingDataOffset();

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
    // really be querying an appropriate absolute offset from `specializedLayout`.
    //
    BindingOffset ordinaryDataBufferOffset = offset;
    SLANG_RETURN_ON_FAIL(_bindOrdinaryDataBufferIfNeeded(
        context,
        /*inout*/ ordinaryDataBufferOffset,
        specializedLayout));
    SLANG_RETURN_ON_FAIL(bindAsValue(context, offset, specializedLayout));

    // Once the state stored in the root shader object itself has been bound,
    // we turn our attention to the entry points and their parameters.
    //
    auto entryPointCount = m_entryPoints.getCount();
    for (Index i = 0; i < entryPointCount; ++i)
    {
        auto entryPoint = m_entryPoints[i];
        auto const& entryPointInfo = specializedLayout->getEntryPoint(i);

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

    for (auto entryPointInfo : layout->getEntryPoints())
    {
        RefPtr<ShaderObjectImpl> entryPoint;
        SLANG_RETURN_ON_FAIL(
            ShaderObjectImpl::create(device, entryPointInfo.layout, entryPoint.writeRef()));
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
    // form that of `compose(specialize(A,X), specialize(B,Y))`, even when both are
    // semantically equivalent programs.
    //
    // Right now we are using the first option: we are first generating a full composition
    // of all the code we plan to use (global scope plus all entry points), and then
    // specializing it to the concatenated specialization arguments for all of that.
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
    // parameters, but their layouts are also independent of one another.
    //
    // Furthermore, in this example, loading another entry point into the system would not
    // require re-computing the layouts (or generated kernel code) for any of the entry points
    // that had already been loaded (in contrast to a compose-then-specialize approach).
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
    RefPtr<RootShaderObjectLayoutImpl> specializedLayout;
    RootShaderObjectLayoutImpl::create(
        getRenderer(),
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
} // namespace d3d11
} // namespace gfx
