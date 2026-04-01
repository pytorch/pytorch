// d3d12-shader-object.cpp
#include "d3d12-shader-object.h"

#include "d3d12-buffer.h"
#include "d3d12-command-encoder.h"
#include "d3d12-device.h"
#include "d3d12-helper-functions.h"
#include "d3d12-resource-views.h"
#include "d3d12-sampler.h"
#include "d3d12-shader-object-layout.h"
#include "d3d12-transient-heap.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

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

// TODO: Change Index to Offset/Size?
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

    m_version++;

    return SLANG_OK;
}

Result ShaderObjectImpl::setObject(ShaderOffset const& offset, IShaderObject* object)
{
    SLANG_RETURN_ON_FAIL(Super::setObject(offset, object));
    if (m_isMutable)
    {
        auto subObjectIndex = getSubObjectIndex(offset);
        if (subObjectIndex >= m_subObjectVersions.getCount())
            m_subObjectVersions.setCount(subObjectIndex + 1);
        m_subObjectVersions[subObjectIndex] = static_cast<ShaderObjectImpl*>(object)->m_version;
        m_version++;
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
    auto samplerImpl = static_cast<SamplerStateImpl*>(sampler);
    ID3D12Device* d3dDevice = static_cast<DeviceImpl*>(getDevice())->m_device;
    d3dDevice->CopyDescriptorsSimple(
        1,
        m_descriptorSet.samplerTable.getCpuHandle(
            bindingRange.baseIndex + (int32_t)offset.bindingArrayIndex),
        samplerImpl->m_descriptor.cpuHandle,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    m_version++;
    return SLANG_OK;
}

Result ShaderObjectImpl::setCombinedTextureSampler(
    ShaderOffset const& offset,
    IResourceView* textureView,
    ISamplerState* sampler)
{
#if 0
    if (offset.bindingRangeIndex < 0)
        return SLANG_E_INVALID_ARG;
    auto layout = getLayout();
    if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
        return SLANG_E_INVALID_ARG;
    auto& bindingRange = layout->getBindingRange(offset.bindingRangeIndex);
    auto resourceViewImpl = static_cast<ResourceViewImpl*>(textureView);
    ID3D12Device* d3dDevice = static_cast<DeviceImpl*>(getDevice())->m_device;
    d3dDevice->CopyDescriptorsSimple(
        1,
        m_resourceHeap.getCpuHandle(
            m_descriptorSet.m_resourceTable +
            bindingRange.binding.offsetInDescriptorTable.resource +
            (int32_t)offset.bindingArrayIndex),
        resourceViewImpl->m_descriptor.cpuHandle,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto samplerImpl = static_cast<SamplerStateImpl*>(sampler);
    d3dDevice->CopyDescriptorsSimple(
        1,
        m_samplerHeap.getCpuHandle(
            m_descriptorSet.m_samplerTable +
            bindingRange.binding.offsetInDescriptorTable.sampler +
            (int32_t)offset.bindingArrayIndex),
        samplerImpl->m_descriptor.cpuHandle,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
#endif
    m_version++;
    return SLANG_OK;
}

Result ShaderObjectImpl::init(
    DeviceImpl* device,
    ShaderObjectLayoutImpl* layout,
    DescriptorHeapReference viewHeap,
    DescriptorHeapReference samplerHeap)
{
    m_device = device;

    m_layout = layout;

    m_cachedTransientHeap = nullptr;
    m_cachedTransientHeapVersion = 0;
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
    size_t uniformSize = layout->getElementTypeLayout()->getSize();
    if (uniformSize)
    {
        m_data.setCount(uniformSize);
        memset(m_data.getBuffer(), 0, uniformSize);
    }
    m_rootArguments.setCount(layout->getOwnUserRootParameterCount());
    memset(
        m_rootArguments.getBuffer(),
        0,
        sizeof(D3D12_GPU_VIRTUAL_ADDRESS) * m_rootArguments.getCount());
    // Each shader object will own CPU descriptor heap memory
    // for any resource or sampler descriptors it might store
    // as part of its value.
    //
    // This allocate includes a reservation for any constant
    // buffer descriptor pertaining to the ordinary data,
    // but does *not* include any descriptors that are managed
    // as part of sub-objects.
    //
    if (auto resourceCount = layout->getResourceSlotCount())
    {
        m_descriptorSet.resourceTable.allocate(viewHeap, resourceCount);

        // We must also ensure that the memory for any resources
        // referenced by descriptors in this object does not get
        // freed while the object is still live.
        //
        // The doubling here is because any buffer resource could
        // have a counter buffer associated with it, which we
        // also need to ensure isn't destroyed prematurely.
        m_boundResources.setCount(resourceCount);
        m_boundCounterResources.setCount(resourceCount);
    }
    if (auto samplerCount = layout->getSamplerSlotCount())
    {
        m_descriptorSet.samplerTable.allocate(samplerHeap, samplerCount);
    }

    // If the layout specifies that we have any sub-objects, then
    // we need to size the array to account for them.
    //
    Index subObjectCount = layout->getSubObjectSlotCount();
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
        for (uint32_t i = 0; i < bindingRangeInfo.count; ++i)
        {
            RefPtr<ShaderObjectImpl> subObject;
            SLANG_RETURN_ON_FAIL(
                ShaderObjectImpl::create(device, subObjectLayout, subObject.writeRef()));
            m_objects[bindingRangeInfo.subObjectIndex + i] = subObject;
        }
    }

    return SLANG_OK;
}

/// Write the uniform/ordinary data of this object into the given `dest` buffer at the given
/// `offset`

Result ShaderObjectImpl::_writeOrdinaryData(
    PipelineCommandEncoder* encoder,
    BufferResourceImpl* buffer,
    Offset offset,
    Size destSize,
    ShaderObjectLayoutImpl* specializedLayout)
{
    auto src = m_data.getBuffer();
    auto srcSize = Size(m_data.getCount());

    SLANG_ASSERT(srcSize <= destSize);

    uploadBufferDataImpl(
        encoder->m_device,
        encoder->m_d3dCmdList,
        encoder->m_transientHeap,
        buffer,
        offset,
        srcSize,
        src);

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

        for (uint32_t i = 0; i < count; ++i)
        {
            auto subObject = m_objects[bindingRangeInfo.subObjectIndex + i];

            RefPtr<ShaderObjectLayoutImpl> subObjectLayout;
            SLANG_RETURN_ON_FAIL(subObject->getSpecializedLayout(subObjectLayout.writeRef()));

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

bool ShaderObjectImpl::shouldAllocateConstantBuffer(TransientResourceHeapImpl* transientHeap)
{
    if (m_isConstantBufferDirty || m_cachedTransientHeap != transientHeap ||
        m_cachedTransientHeapVersion != transientHeap->getVersion())
    {
        return true;
    }
    return false;
}

/// Ensure that the `m_ordinaryDataBuffer` has been created, if it is needed

Result ShaderObjectImpl::_ensureOrdinaryDataBufferCreatedIfNeeded(
    PipelineCommandEncoder* encoder,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // If data has been changed since last allocation/filling of constant buffer,
    // we will need to allocate a new one.
    //
    if (!shouldAllocateConstantBuffer(encoder->m_transientHeap))
    {
        return SLANG_OK;
    }
    m_isConstantBufferDirty = false;
    m_cachedTransientHeap = encoder->m_transientHeap;
    m_cachedTransientHeapVersion = encoder->m_transientHeap->getVersion();

    // Computing the size of the ordinary data buffer is *not* just as simple
    // as using the size of the `m_ordinayData` array that we store. The reason
    // for the added complexity is that interface-type fields may lead to the
    // storage being specialized such that it needs extra appended data to
    // store the concrete values that logically belong in those interface-type
    // fields but wouldn't fit in the fixed-size allocation we gave them.
    //
    m_constantBufferSize = specializedLayout->getTotalOrdinaryDataSize();
    if (m_constantBufferSize == 0)
    {
        return SLANG_OK;
    }

    // Once we have computed how large the buffer should be, we can allocate
    // it from the transient resource heap.
    //
    auto alignedConstantBufferSize = D3DUtil::calcAligned(m_constantBufferSize, 256);
    SLANG_RETURN_ON_FAIL(encoder->m_commandBuffer->m_transientHeap->allocateConstantBuffer(
        alignedConstantBufferSize,
        m_constantBufferWeakPtr,
        m_constantBufferOffset));

    // Once the buffer is allocated, we can use `_writeOrdinaryData` to fill it in.
    //
    // Note that `_writeOrdinaryData` is potentially recursive in the case
    // where this object contains interface/existential-type fields, so we
    // don't need or want to inline it into this call site.
    //
    SLANG_RETURN_ON_FAIL(_writeOrdinaryData(
        encoder,
        static_cast<BufferResourceImpl*>(m_constantBufferWeakPtr),
        m_constantBufferOffset,
        m_constantBufferSize,
        specializedLayout));

    {
        // We also create and store a descriptor for our root constant buffer
        // into the descriptor table allocation that was reserved for them.
        //
        // We always know that the ordinary data buffer will be the first descriptor
        // in the table of resource views.
        //
        auto descriptorTable = m_descriptorSet.resourceTable;
        D3D12_CONSTANT_BUFFER_VIEW_DESC viewDesc = {};
        viewDesc.BufferLocation = static_cast<BufferResourceImpl*>(m_constantBufferWeakPtr)
                                      ->m_resource.getResource()
                                      ->GetGPUVirtualAddress() +
                                  m_constantBufferOffset;
        viewDesc.SizeInBytes = (UINT)alignedConstantBufferSize;
        encoder->m_device->CreateConstantBufferView(&viewDesc, descriptorTable.getCpuHandle());
    }

    return SLANG_OK;
}

void ShaderObjectImpl::updateSubObjectsRecursive()
{
    if (!m_isMutable)
        return;
    auto& subObjectRanges = getLayout()->getSubObjectRanges();
    for (Slang::Index subObjectRangeIndex = 0; subObjectRangeIndex < subObjectRanges.getCount();
         subObjectRangeIndex++)
    {
        auto const& subObjectRange = subObjectRanges[subObjectRangeIndex];
        auto const& bindingRange = getLayout()->getBindingRange(subObjectRange.bindingRangeIndex);
        Slang::Index count = bindingRange.count;

        for (Slang::Index subObjectIndexInRange = 0; subObjectIndexInRange < count;
             subObjectIndexInRange++)
        {
            Slang::Index objectIndex = bindingRange.subObjectIndex + subObjectIndexInRange;
            auto subObject = m_objects[objectIndex].Ptr();
            if (!subObject)
                continue;
            subObject->updateSubObjectsRecursive();
            if (m_subObjectVersions.getCount() > objectIndex &&
                m_subObjectVersions[objectIndex] != m_objects[objectIndex]->m_version)
            {
                ShaderOffset offset;
                offset.bindingRangeIndex = (GfxIndex)subObjectRange.bindingRangeIndex;
                offset.bindingArrayIndex = (GfxIndex)subObjectIndexInRange;
                setObject(offset, subObject);
            }
        }
    }
}

static void bindPendingTables(BindingContext* context)
{
    for (auto& binding : *context->pendingTableBindings)
    {
        context->submitter->setRootDescriptorTable(binding.rootIndex, binding.handle);
    }
}

/// Prepare to bind this object as a parameter block.
///
/// This involves allocating and binding any descriptor tables necessary
/// to to store the state of the object. The function returns a descriptor
/// set formed from any table(s) allocated. In addition, the `ioOffset`
/// parameter will be adjusted to be correct for binding values into
/// the resulting descriptor set.
///
/// Returns:
///   SLANG_OK when successful,
///   SLANG_E_OUT_OF_MEMORY when descriptor heap is full.
///

Result ShaderObjectImpl::prepareToBindAsParameterBlock(
    BindingContext* context,
    BindingOffset& ioOffset,
    ShaderObjectLayoutImpl* specializedLayout,
    DescriptorSet& outDescriptorSet)
{
    auto transientHeap = context->transientHeap;
    auto submitter = context->submitter;

    // When writing into the new descriptor set, resource and sampler
    // descriptors will need to start at index zero in the respective
    // tables.
    //
    ioOffset.resource = 0;
    ioOffset.sampler = 0;

    // The index of the next root parameter to bind will be maintained,
    // but needs to be incremented by the number of descriptor tables
    // we allocate (zero or one resource table and zero or one sampler
    // table).
    //
    auto& rootParamIndex = ioOffset.rootParam;

    if (auto descriptorCount = specializedLayout->getTotalResourceDescriptorCount())
    {
        // There is a non-zero number of resource descriptors needed,
        // so we will allocate a table out of the appropriate heap,
        // and store it into the appropriate part of `descriptorSet`.
        //
        auto descriptorHeap = &transientHeap->getCurrentViewHeap();
        auto& table = outDescriptorSet.resourceTable;

        // Allocate the table.
        //
        if (!table.allocate(descriptorHeap, descriptorCount))
        {
            context->outOfMemoryHeap = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            return SLANG_E_OUT_OF_MEMORY;
        }

        // Bind the table to the pipeline, consuming the next available
        // root parameter.
        //
        auto tableRootParamIndex = rootParamIndex++;
        context->pendingTableBindings->add(
            PendingDescriptorTableBinding{tableRootParamIndex, table.getGpuHandle()});
    }
    if (auto descriptorCount = specializedLayout->getTotalSamplerDescriptorCount())
    {
        // There is a non-zero number of sampler descriptors needed,
        // so we will allocate a table out of the appropriate heap,
        // and store it into the appropriate part of `descriptorSet`.
        //
        auto descriptorHeap = &transientHeap->getCurrentSamplerHeap();
        auto& table = outDescriptorSet.samplerTable;

        // Allocate the table.
        //
        if (!table.allocate(descriptorHeap, descriptorCount))
        {
            context->outOfMemoryHeap = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            return SLANG_E_OUT_OF_MEMORY;
        }

        // Bind the table to the pipeline, consuming the next available
        // root parameter.
        //
        auto tableRootParamIndex = rootParamIndex++;
        context->pendingTableBindings->add(
            PendingDescriptorTableBinding{tableRootParamIndex, table.getGpuHandle()});
    }

    return SLANG_OK;
}

bool ShaderObjectImpl::checkIfCachedDescriptorSetIsValidRecursive(BindingContext* context)
{
    if (shouldAllocateConstantBuffer(context->transientHeap))
        return false;
    if (m_isMutable && m_version != m_cachedGPUDescriptorSetVersion)
        return false;
    if (m_cachedGPUDescriptorSet.resourceTable.getDescriptorCount() != 0 &&
        m_cachedGPUDescriptorSet.resourceTable.m_heap.ptr.linearHeap->getHeap() !=
            m_cachedTransientHeap->getCurrentViewHeap().getHeap())
        return false;
    if (m_cachedGPUDescriptorSet.samplerTable.getDescriptorCount() != 0 &&
        m_cachedGPUDescriptorSet.samplerTable.m_heap.ptr.linearHeap->getHeap() !=
            m_cachedTransientHeap->getCurrentSamplerHeap().getHeap())
        return false;

    auto& subObjectRanges = getLayout()->getSubObjectRanges();
    for (Slang::Index subObjectRangeIndex = 0; subObjectRangeIndex < subObjectRanges.getCount();
         subObjectRangeIndex++)
    {
        auto const& subObjectRange = subObjectRanges[subObjectRangeIndex];
        auto const& bindingRange = getLayout()->getBindingRange(subObjectRange.bindingRangeIndex);
        if (bindingRange.bindingType != slang::BindingType::ParameterBlock)
            continue;
        Slang::Index count = bindingRange.count;

        for (Slang::Index subObjectIndexInRange = 0; subObjectIndexInRange < count;
             subObjectIndexInRange++)
        {
            Slang::Index objectIndex = bindingRange.subObjectIndex + subObjectIndexInRange;
            auto subObject = m_objects[objectIndex].Ptr();
            if (!subObject)
                continue;
            if (subObject->checkIfCachedDescriptorSetIsValidRecursive(context))
                return false;
        }
    }
    return true;
}

/// Bind this object as a `ParameterBlock<X>`

Result ShaderObjectImpl::bindAsParameterBlock(
    BindingContext* context,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    if (checkIfCachedDescriptorSetIsValidRecursive(context))
    {
        // If we already have a valid gpu descriptor table in the current
        // heap, bind it.
        auto rootParamIndex = offset.rootParam;
        if (m_cachedGPUDescriptorSet.resourceTable.getDescriptorCount())
        {
            auto tableRootParamIndex = rootParamIndex++;
            context->submitter->setRootDescriptorTable(
                tableRootParamIndex,
                m_cachedGPUDescriptorSet.resourceTable.getGpuHandle());
        }
        if (m_cachedGPUDescriptorSet.samplerTable.getDescriptorCount())
        {
            auto tableRootParamIndex = rootParamIndex++;
            context->submitter->setRootDescriptorTable(
                tableRootParamIndex,
                m_cachedGPUDescriptorSet.samplerTable.getGpuHandle());
        }
        return SLANG_OK;
    }

    // The first step to binding an object as a parameter block is to allocate a descriptor
    // set (consisting of zero or one resource descriptor table and zero or one sampler
    // descriptor table) to represent its values.
    //
    BindingOffset subOffset = offset;
    ShortList<PendingDescriptorTableBinding> pendingTableBindings;
    auto oldPendingTableBindings = context->pendingTableBindings;
    context->pendingTableBindings = &pendingTableBindings;

    SLANG_RETURN_ON_FAIL(prepareToBindAsParameterBlock(
        context,
        /* inout */ subOffset,
        specializedLayout,
        m_cachedGPUDescriptorSet));

    // Next we bind the object into that descriptor set as if it were being used
    // as a `ConstantBuffer<X>`.
    //
    SLANG_RETURN_ON_FAIL(
        bindAsConstantBuffer(context, m_cachedGPUDescriptorSet, subOffset, specializedLayout));

    bindPendingTables(context);
    context->pendingTableBindings = oldPendingTableBindings;

    m_cachedGPUDescriptorSetVersion = m_version;
    return SLANG_OK;
}

/// Bind this object as a `ConstantBuffer<X>`

Result ShaderObjectImpl::bindAsConstantBuffer(
    BindingContext* context,
    DescriptorSet const& descriptorSet,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // If we are to bind as a constant buffer we first need to ensure that
    // the ordinary data buffer is created, if this object needs one.
    //
    SLANG_RETURN_ON_FAIL(
        _ensureOrdinaryDataBufferCreatedIfNeeded(context->encoder, specializedLayout));

    // Next, we need to bind all of the resource descriptors for this object
    // (including any ordinary data buffer) into the provided `descriptorSet`.
    //
    auto resourceCount = specializedLayout->getResourceSlotCount();
    if (resourceCount)
    {
        auto& dstTable = descriptorSet.resourceTable;
        auto& srcTable = m_descriptorSet.resourceTable;

        context->device->m_device->CopyDescriptorsSimple(
            UINT(resourceCount),
            dstTable.getCpuHandle(offset.resource),
            srcTable.getCpuHandle(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    // Finally, we delegate to `_bindImpl` to bind samplers and sub-objects,
    // since the logic is shared with the `bindAsValue()` case below.
    //
    SLANG_RETURN_ON_FAIL(_bindImpl(context, descriptorSet, offset, specializedLayout));
    return SLANG_OK;
}

/// Bind this object as a value (for an interface-type parameter)

Result ShaderObjectImpl::bindAsValue(
    BindingContext* context,
    DescriptorSet const& descriptorSet,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // When binding a value for an interface-type field we do *not* want
    // to bind a buffer for the ordinary data (if there is any) because
    // ordinary data for interface-type fields gets allocated into the
    // parent object's ordinary data buffer.
    //
    // This CPU-memory descriptor table that holds resource descriptors
    // will have already been allocated to have space for an ordinary data
    // buffer (if needed), so we need to take care to skip over that
    // descriptor when copying descriptors from the CPU-memory set
    // to the GPU-memory `descriptorSet`.
    //
    auto skipResourceCount = specializedLayout->getOrdinaryDataBufferCount();
    auto resourceCount = specializedLayout->getResourceSlotCount() - skipResourceCount;
    if (resourceCount)
    {
        auto& dstTable = descriptorSet.resourceTable;
        auto& srcTable = m_descriptorSet.resourceTable;

        context->device->m_device->CopyDescriptorsSimple(
            UINT(resourceCount),
            dstTable.getCpuHandle(offset.resource),
            srcTable.getCpuHandle(skipResourceCount),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    // Finally, we delegate to `_bindImpl` to bind samplers and sub-objects,
    // since the logic is shared with the `bindAsConstantBuffer()` case above.
    //
    // Note: Just like we had to do some subtle handling of the ordinary data buffer
    // above, here we need to contend with the fact that the `offset.resource` fields
    // computed for sub-object ranges were baked to take the ordinary data buffer
    // into account, so that if `skipResourceCount` is non-zero then they are all
    // too high by `skipResourceCount`.
    //
    // We will address the problem here by computing a modified offset that adjusts
    // for the ordinary data buffer that we have not bound after all.
    //
    BindingOffset subOffset = offset;
    subOffset.resource -= skipResourceCount;
    SLANG_RETURN_ON_FAIL(_bindImpl(context, descriptorSet, subOffset, specializedLayout));
    return SLANG_OK;
}

/// Shared logic for `bindAsConstantBuffer()` and `bindAsValue()`

Result ShaderObjectImpl::_bindImpl(
    BindingContext* context,
    DescriptorSet const& descriptorSet,
    BindingOffset const& offset,
    ShaderObjectLayoutImpl* specializedLayout)
{
    // We start by binding all the sampler decriptors, if needed.
    //
    // Note: resource descriptors were handled in either `bindAsConstantBuffer()`
    // or `bindAsValue()` before calling into `_bindImpl()`.
    //
    if (auto samplerCount = specializedLayout->getSamplerSlotCount())
    {
        auto& dstTable = descriptorSet.samplerTable;
        auto& srcTable = m_descriptorSet.samplerTable;

        context->device->m_device->CopyDescriptorsSimple(
            UINT(samplerCount),
            dstTable.getCpuHandle(offset.sampler),
            srcTable.getCpuHandle(),
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    }

    // Next we iterate over the sub-object ranges and bind anything they require.
    //
    auto& subObjectRanges = specializedLayout->getSubObjectRanges();
    auto subObjectRangeCount = subObjectRanges.getCount();
    for (Index i = 0; i < subObjectRangeCount; i++)
    {
        auto& subObjectRange = specializedLayout->getSubObjectRange(i);
        auto& bindingRange = specializedLayout->getBindingRange(subObjectRange.bindingRangeIndex);
        auto subObjectIndex = bindingRange.subObjectIndex;
        auto subObjectLayout = subObjectRange.layout.Ptr();

        BindingOffset rangeOffset = offset;
        rangeOffset += subObjectRange.offset;

        BindingOffset rangeStride = subObjectRange.stride;

        switch (bindingRange.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
            {
                auto objOffset = rangeOffset;
                for (uint32_t j = 0; j < bindingRange.count; j++)
                {
                    auto& object = m_objects[subObjectIndex + j];
                    SLANG_RETURN_ON_FAIL(object->bindAsConstantBuffer(
                        context,
                        descriptorSet,
                        objOffset,
                        subObjectLayout));
                    objOffset += rangeStride;
                }
            }
            break;

        case slang::BindingType::ParameterBlock:
            {
                auto objOffset = rangeOffset;
                for (uint32_t j = 0; j < bindingRange.count; j++)
                {
                    auto& object = m_objects[subObjectIndex + j];
                    SLANG_RETURN_ON_FAIL(
                        object->bindAsParameterBlock(context, objOffset, subObjectLayout));
                    objOffset += rangeStride;
                }
            }
            break;

        case slang::BindingType::ExistentialValue:
            if (subObjectLayout)
            {
                auto objOffset = rangeOffset;
                for (uint32_t j = 0; j < bindingRange.count; j++)
                {
                    auto& object = m_objects[subObjectIndex + j];
                    SLANG_RETURN_ON_FAIL(
                        object->bindAsValue(context, descriptorSet, objOffset, subObjectLayout));
                    objOffset += rangeStride;
                }
            }
            break;
        }
    }

    return SLANG_OK;
}

Result ShaderObjectImpl::bindRootArguments(BindingContext* context, uint32_t& index)
{
    auto layoutImpl = getLayout();
    for (Index i = 0; i < m_rootArguments.getCount(); i++)
    {
        switch (layoutImpl->getRootParameterInfo(i).type)
        {
        case IResourceView::Type::ShaderResource:
        case IResourceView::Type::AccelerationStructure:
            context->submitter->setRootSRV(index, m_rootArguments[i]);
            break;
        case IResourceView::Type::UnorderedAccess:
            context->submitter->setRootUAV(index, m_rootArguments[i]);
            break;
        default:
            continue;
        }
        index++;
    }
    for (auto& subObject : m_objects)
    {
        if (subObject)
        {
            SLANG_RETURN_ON_FAIL(subObject->bindRootArguments(context, index));
        }
    }
    return SLANG_OK;
}

/// Get the layout of this shader object with specialization arguments considered
///
/// This operation should only be called after the shader object has been
/// fully filled in and finalized.
///

Result ShaderObjectImpl::getSpecializedLayout(ShaderObjectLayoutImpl** outLayout)
{
    if (!m_specializedLayout)
    {
        SLANG_RETURN_ON_FAIL(_createSpecializedLayout(m_specializedLayout.writeRef()));
    }
    returnRefPtr(outLayout, m_specializedLayout);
    return SLANG_OK;
}

/// Create the layout for this shader object with specialization arguments considered
///
/// This operation is virtual so that it can be customized by `RootShaderObject`.
///

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

Result ShaderObjectImpl::setResource(ShaderOffset const& offset, IResourceView* resourceView)
{
    if (offset.bindingRangeIndex < 0)
        return SLANG_E_INVALID_ARG;
    auto layout = getLayout();
    if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
        return SLANG_E_INVALID_ARG;

    m_version++;

    ID3D12Device* d3dDevice = static_cast<DeviceImpl*>(getDevice())->m_device;

    auto& bindingRange = layout->getBindingRange(offset.bindingRangeIndex);

    if (bindingRange.isRootParameter && resourceView)
    {
        auto& rootArg = m_rootArguments[bindingRange.baseIndex];
        switch (resourceView->getViewDesc()->type)
        {
        case IResourceView::Type::AccelerationStructure:
            {
                auto resourceViewImpl = static_cast<AccelerationStructureImpl*>(resourceView);
                rootArg = resourceViewImpl->getDeviceAddress();
            }
            break;
        case IResourceView::Type::ShaderResource:
        case IResourceView::Type::UnorderedAccess:
            {
                auto resourceViewImpl = static_cast<ResourceViewImpl*>(resourceView);
                if (resourceViewImpl->m_resource->isBuffer())
                {
                    rootArg = static_cast<BufferResourceImpl*>(resourceViewImpl->m_resource.Ptr())
                                  ->getDeviceAddress();
                }
                else
                {
                    getDebugCallback()->handleMessage(
                        DebugMessageType::Error,
                        DebugMessageSource::Layer,
                        "The shader parameter at the specified offset is a root parameter, and "
                        "therefore can only be a buffer view.");
                    return SLANG_FAIL;
                }
            }
            break;
        }
        return SLANG_OK;
    }

    if (resourceView == nullptr)
    {
        if (!bindingRange.isRootParameter)
        {
            // Create null descriptor for the binding.
            auto destDescriptor = m_descriptorSet.resourceTable.getCpuHandle(
                bindingRange.baseIndex + (int32_t)offset.bindingArrayIndex);
            return createNullDescriptor(d3dDevice, destDescriptor, bindingRange);
        }
        return SLANG_OK;
    }

    ResourceViewInternalImpl* internalResourceView = nullptr;
    auto resourceViewImpl = static_cast<ResourceViewImpl*>(resourceView);

    switch (resourceView->getViewDesc()->type)
    {
#if SLANG_GFX_HAS_DXR_SUPPORT
    case IResourceView::Type::AccelerationStructure:
        {
            auto asImpl = static_cast<AccelerationStructureImpl*>(resourceView);
            // Hold a reference to the resource to prevent its destruction.
            m_boundResources[bindingRange.baseIndex + offset.bindingArrayIndex] = asImpl->m_buffer;
            internalResourceView = asImpl;
        }
        break;
#endif
    default:
        {
            // Hold a reference to the resource to prevent its destruction.
            const auto resourceOffset = bindingRange.baseIndex + offset.bindingArrayIndex;
            m_boundResources[resourceOffset] = resourceViewImpl->m_resource;
            m_boundCounterResources[resourceOffset] = resourceViewImpl->m_counterResource;
            internalResourceView = resourceViewImpl;
        }
        break;
    }

    auto descriptorSlotIndex = bindingRange.baseIndex + (int32_t)offset.bindingArrayIndex;
    D3D12Descriptor srcDescriptor = internalResourceView->m_descriptor;

    // Buffer descriptors are created on demand.
    if (!srcDescriptor.cpuHandle.ptr)
    {
        SLANG_RETURN_ON_FAIL(internalResourceView->getBufferDescriptorForBinding(
            static_cast<DeviceImpl*>(m_device.get()),
            resourceViewImpl,
            bindingRange.bufferElementStride,
            srcDescriptor));
    }

    if (srcDescriptor.cpuHandle.ptr)
    {
        d3dDevice->CopyDescriptorsSimple(
            1,
            m_descriptorSet.resourceTable.getCpuHandle(
                bindingRange.baseIndex + (int32_t)offset.bindingArrayIndex),
            srcDescriptor.cpuHandle,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    else
    {
        getDebugCallback()->handleMessage(
            DebugMessageType::Error,
            DebugMessageSource::Layer,
            "IShaderObject::setResource: the resource view cannot be set to this shader parameter. "
            "A possible reason is that the view is too large to be supported by D3D12.");
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

Result ShaderObjectImpl::create(
    DeviceImpl* device,
    ShaderObjectLayoutImpl* layout,
    ShaderObjectImpl** outShaderObject)
{
    auto object = RefPtr<ShaderObjectImpl>(new ShaderObjectImpl());
    SLANG_RETURN_ON_FAIL(
        object->init(device, layout, device->m_cpuViewHeap.Ptr(), device->m_cpuSamplerHeap.Ptr()));
    returnRefPtrMove(outShaderObject, object);
    return SLANG_OK;
}

ShaderObjectImpl::~ShaderObjectImpl()
{
    m_descriptorSet.freeIfSupported();
}

RootShaderObjectLayoutImpl* RootShaderObjectImpl::getLayout()
{
    return static_cast<RootShaderObjectLayoutImpl*>(m_layout.Ptr());
}

GfxCount RootShaderObjectImpl::getEntryPointCount()
{
    return (GfxCount)m_entryPoints.getCount();
}

SlangResult RootShaderObjectImpl::getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint)
{
    returnComPtr(outEntryPoint, m_entryPoints[index]);
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

    if (diagnosticBlob && diagnosticBlob->getBufferSize())
    {
        getDebugCallback()->handleMessage(
            SLANG_FAILED(result) ? DebugMessageType::Error : DebugMessageType::Info,
            DebugMessageSource::Layer,
            (const char*)diagnosticBlob->getBufferPointer());
    }

    if (SLANG_FAILED(result))
        return result;

    ComPtr<ID3DBlob> d3dDiagnosticBlob;
    auto slangSpecializedLayout = specializedComponentType->getLayout();
    RefPtr<RootShaderObjectLayoutImpl> specializedLayout;
    auto rootLayoutResult = RootShaderObjectLayoutImpl::create(
        static_cast<DeviceImpl*>(getRenderer()),
        specializedComponentType,
        slangSpecializedLayout,
        specializedLayout.writeRef(),
        d3dDiagnosticBlob.writeRef());

    if (SLANG_FAILED(rootLayoutResult))
    {
        return rootLayoutResult;
    }

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

Result RootShaderObjectImpl::copyFrom(IShaderObject* object, ITransientResourceHeap* transientHeap)
{
    if (auto srcObj = dynamic_cast<MutableRootShaderObjectImpl*>(object))
    {
        *this = *srcObj;
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

Result RootShaderObjectImpl::bindAsRoot(
    BindingContext* context,
    RootShaderObjectLayoutImpl* specializedLayout)
{
    // Pull updates from sub-objects when this is a mutable root shader object.
    updateSubObjectsRecursive();

    // A root shader object always binds as if it were a parameter block,
    // insofar as it needs to allocate a descriptor set to hold the bindings
    // for its own state and any sub-objects.
    //
    // Note: We do not direclty use `bindAsParameterBlock` here because we also
    // need to bind the entry points into the same descriptor set that is
    // being used for the root object.

    ShortList<PendingDescriptorTableBinding> pendingTableBindings;
    auto oldPendingTableBindings = context->pendingTableBindings;
    context->pendingTableBindings = &pendingTableBindings;

    BindingOffset rootOffset;

    // Bind all root parameters first.
    Super::bindRootArguments(context, rootOffset.rootParam);

    DescriptorSet descriptorSet;
    SLANG_RETURN_ON_FAIL(prepareToBindAsParameterBlock(
        context,
        /* inout */ rootOffset,
        specializedLayout,
        descriptorSet));

    SLANG_RETURN_ON_FAIL(
        Super::bindAsConstantBuffer(context, descriptorSet, rootOffset, specializedLayout));

    auto entryPointCount = m_entryPoints.getCount();
    for (Index i = 0; i < entryPointCount; ++i)
    {
        auto entryPoint = m_entryPoints[i];
        auto& entryPointInfo = specializedLayout->getEntryPoint(i);

        auto entryPointOffset = rootOffset;
        entryPointOffset += entryPointInfo.offset;

        entryPoint->updateSubObjectsRecursive();

        SLANG_RETURN_ON_FAIL(entryPoint->bindAsConstantBuffer(
            context,
            descriptorSet,
            entryPointOffset,
            entryPointInfo.layout));
    }

    bindPendingTables(context);
    context->pendingTableBindings = oldPendingTableBindings;

    return SLANG_OK;
}

Result RootShaderObjectImpl::resetImpl(
    DeviceImpl* device,
    RootShaderObjectLayoutImpl* layout,
    DescriptorHeapReference viewHeap,
    DescriptorHeapReference samplerHeap,
    bool isMutable)
{
    SLANG_RETURN_ON_FAIL(Super::init(device, layout, viewHeap, samplerHeap));
    m_isMutable = isMutable;
    m_specializedLayout = nullptr;
    m_entryPoints.clear();
    for (auto entryPointInfo : layout->getEntryPoints())
    {
        RefPtr<ShaderObjectImpl> entryPoint;
        SLANG_RETURN_ON_FAIL(
            ShaderObjectImpl::create(device, entryPointInfo.layout, entryPoint.writeRef()));
        entryPoint->m_isMutable = isMutable;
        m_entryPoints.add(entryPoint);
    }
    return SLANG_OK;
}

Result RootShaderObjectImpl::reset(
    DeviceImpl* device,
    RootShaderObjectLayoutImpl* layout,
    TransientResourceHeapImpl* heap)
{
    return resetImpl(
        device,
        layout,
        &heap->m_stagingCpuViewHeap,
        &heap->m_stagingCpuSamplerHeap,
        false);
}

} // namespace d3d12
} // namespace gfx
