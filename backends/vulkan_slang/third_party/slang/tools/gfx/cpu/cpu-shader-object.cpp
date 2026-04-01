// cpu-shader-object.cpp
#include "cpu-shader-object.h"

#include "cpu-buffer.h"
#include "cpu-resource-views.h"
#include "cpu-shader-object-layout.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

Index CPUShaderObjectData::getCount()
{
    return m_ordinaryData.getCount();
}

void CPUShaderObjectData::setCount(Index count)
{
    m_ordinaryData.setCount(count);
}

char* CPUShaderObjectData::getBuffer()
{
    return m_ordinaryData.getBuffer();
}

CPUShaderObjectData::~CPUShaderObjectData()
{
    // m_bufferResource's data is managed by m_ordinaryData so we
    // set it to null to prevent m_bufferResource from freeing it.
    if (m_bufferResource)
        m_bufferResource->m_data = nullptr;
}

/// Returns a StructuredBuffer resource view for GPU access into the buffer content.
/// Creates a StructuredBuffer resource if it has not been created.
ResourceViewBase* CPUShaderObjectData::getResourceView(
    RendererBase* device,
    slang::TypeLayoutReflection* elementLayout,
    slang::BindingType bindingType)
{
    SLANG_UNUSED(device);
    if (!m_bufferResource)
    {
        IBufferResource::Desc desc = {};
        desc.type = IResource::Type::Buffer;
        desc.elementSize = (int)elementLayout->getSize();
        m_bufferResource = new BufferResourceImpl(desc);

        IResourceView::Desc viewDesc = {};
        viewDesc.type = IResourceView::Type::UnorderedAccess;
        viewDesc.format = Format::Unknown;
        m_bufferView = new BufferResourceViewImpl(viewDesc, m_bufferResource);
    }
    m_bufferResource->getDesc()->sizeInBytes = m_ordinaryData.getCount();
    m_bufferResource->m_data = m_ordinaryData.getBuffer();
    return m_bufferView.Ptr();
}

Result ShaderObjectImpl::init(IDevice* device, ShaderObjectLayoutImpl* typeLayout)
{
    m_layout = typeLayout;

    // If the layout tells us that there is any uniform data,
    // then we need to allocate a constant buffer to hold that data.
    //
    // TODO: Do we need to allocate a shadow copy for use from
    // the CPU?
    //
    // TODO: When/where do we bind this constant buffer into
    // a descriptor set for later use?
    //
    auto slangLayout = getLayout()->getElementTypeLayout();
    size_t uniformSize = slangLayout->getSize();
    m_data.setCount(uniformSize);

    // If the layout specifies that we have any resources or sub-objects,
    // then we need to size the appropriate arrays to account for them.
    //
    // Note: the counts here are the *total* number of resources/sub-objects
    // and not just the number of resource/sub-object ranges.
    //
    m_resources.setCount(typeLayout->getResourceCount());
    m_objects.setCount(typeLayout->getSubObjectCount());

    for (auto subObjectRange : getLayout()->subObjectRanges)
    {
        RefPtr<ShaderObjectLayoutImpl> subObjectLayout = subObjectRange.layout;

        // In the case where the sub-object range represents an
        // existential-type leaf field (e.g., an `IBar`), we
        // cannot pre-allocate the object(s) to go into that
        // range, since we can't possibly know what to allocate
        // at this point.
        //
        if (!subObjectLayout)
            continue;
        auto _debugname = subObjectLayout->getElementTypeLayout()->getName();

        //
        // Otherwise, we will allocate a sub-object to fill
        // in each entry in this range, based on the layout
        // information we already have.

        auto& bindingRangeInfo = getLayout()->m_bindingRanges[subObjectRange.bindingRangeIndex];
        for (Index i = 0; i < bindingRangeInfo.count; ++i)
        {
            RefPtr<ShaderObjectImpl> subObject = new ShaderObjectImpl();
            SLANG_RETURN_ON_FAIL(subObject->init(device, subObjectLayout));

            ShaderOffset offset;
            offset.uniformOffset = bindingRangeInfo.uniformOffset + sizeof(void*) * i;
            offset.bindingRangeIndex = (GfxIndex)subObjectRange.bindingRangeIndex;
            offset.bindingArrayIndex = (GfxIndex)i;

            SLANG_RETURN_ON_FAIL(setObject(offset, subObject));
        }
    }
    return SLANG_OK;
}

SLANG_NO_THROW GfxCount SLANG_MCALL ShaderObjectImpl::getEntryPointCount()
{
    return 0;
}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint)
{
    *outEntryPoint = nullptr;
    return SLANG_OK;
}

SLANG_NO_THROW const void* SLANG_MCALL ShaderObjectImpl::getRawData()
{
    return m_data.getBuffer();
}

SLANG_NO_THROW size_t SLANG_MCALL ShaderObjectImpl::getSize()
{
    return (size_t)m_data.getCount();
}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::setData(ShaderOffset const& offset, void const* data, size_t size)
{
    size = Math::Min(size, size_t(m_data.getCount() - offset.uniformOffset));
    memcpy((char*)m_data.getBuffer() + offset.uniformOffset, data, size);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::setResource(ShaderOffset const& offset, IResourceView* inView)
{
    auto layout = getLayout();

    auto bindingRangeIndex = offset.bindingRangeIndex;
    SLANG_ASSERT(bindingRangeIndex >= 0);
    SLANG_ASSERT(bindingRangeIndex < layout->m_bindingRanges.getCount());

    auto& bindingRange = layout->m_bindingRanges[bindingRangeIndex];
    auto viewIndex = bindingRange.baseIndex + offset.bindingArrayIndex;


    auto view = static_cast<ResourceViewImpl*>(inView);
    m_resources[viewIndex] = view;

    switch (view->getViewKind())
    {
    case ResourceViewImpl::Kind::Texture:
        {
            auto textureView = static_cast<TextureResourceViewImpl*>(view);

            slang_prelude::IRWTexture* textureObj = textureView;
            SLANG_RETURN_ON_FAIL(setData(offset, &textureObj, sizeof(textureObj)));
        }
        break;

    case ResourceViewImpl::Kind::Buffer:
        {
            auto bufferView = static_cast<BufferResourceViewImpl*>(view);
            auto buffer = bufferView->getBuffer();
            auto desc = *buffer->getDesc();

            void* dataPtr = buffer->m_data;
            size_t size = desc.sizeInBytes;
            if (desc.elementSize > 1)
                size /= desc.elementSize;

            auto ptrOffset = offset;
            SLANG_RETURN_ON_FAIL(setData(ptrOffset, &dataPtr, sizeof(dataPtr)));

            auto sizeOffset = offset;
            sizeOffset.uniformOffset += sizeof(dataPtr);
            SLANG_RETURN_ON_FAIL(setData(sizeOffset, &size, sizeof(size)));
        }
        break;
    }

    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::setObject(ShaderOffset const& offset, IShaderObject* object)
{
    SLANG_RETURN_ON_FAIL(Super::setObject(offset, object));

    auto bindingRangeIndex = offset.bindingRangeIndex;
    auto& bindingRange = getLayout()->m_bindingRanges[bindingRangeIndex];

    ShaderObjectImpl* subObject = static_cast<ShaderObjectImpl*>(object);

    switch (bindingRange.bindingType)
    {
    default:
        {
            void* bufferPtr = subObject->m_data.getBuffer();
            SLANG_RETURN_ON_FAIL(setData(offset, &bufferPtr, sizeof(void*)));
        }
        break;
    case slang::BindingType::ExistentialValue:
    case slang::BindingType::RawBuffer:
    case slang::BindingType::MutableRawBuffer:
        break;
    }
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
ShaderObjectImpl::setSampler(ShaderOffset const& offset, ISamplerState* sampler)
{
    SLANG_UNUSED(sampler);
    SLANG_UNUSED(offset);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL ShaderObjectImpl::setCombinedTextureSampler(
    ShaderOffset const& offset,
    IResourceView* textureView,
    ISamplerState* sampler)
{
    SLANG_UNUSED(sampler);
    setResource(offset, textureView);
    return SLANG_OK;
}

char* ShaderObjectImpl::getDataBuffer()
{
    return m_data.getBuffer();
}

EntryPointLayoutImpl* EntryPointShaderObjectImpl::getLayout()
{
    return static_cast<EntryPointLayoutImpl*>(m_layout.Ptr());
}

SLANG_NO_THROW uint32_t SLANG_MCALL RootShaderObjectImpl::addRef()
{
    return 1;
}

SLANG_NO_THROW uint32_t SLANG_MCALL RootShaderObjectImpl::release()
{
    return 1;
}

Result RootShaderObjectImpl::init(IDevice* device, RootShaderObjectLayoutImpl* programLayout)
{
    SLANG_RETURN_ON_FAIL(ShaderObjectImpl::init(device, programLayout));
    for (auto& entryPoint : programLayout->m_entryPointLayouts)
    {
        RefPtr<EntryPointShaderObjectImpl> object = new EntryPointShaderObjectImpl();
        SLANG_RETURN_ON_FAIL(object->init(device, entryPoint));
        m_entryPoints.add(object);
    }
    return SLANG_OK;
}

RootShaderObjectLayoutImpl* RootShaderObjectImpl::getLayout()
{
    return static_cast<RootShaderObjectLayoutImpl*>(m_layout.Ptr());
}

EntryPointShaderObjectImpl* RootShaderObjectImpl::getEntryPoint(Index index)
{
    return m_entryPoints[index];
}

SLANG_NO_THROW GfxCount SLANG_MCALL RootShaderObjectImpl::getEntryPointCount()
{
    return (GfxCount)m_entryPoints.getCount();
}

SLANG_NO_THROW Result SLANG_MCALL
RootShaderObjectImpl::getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint)
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

} // namespace cpu
} // namespace gfx
