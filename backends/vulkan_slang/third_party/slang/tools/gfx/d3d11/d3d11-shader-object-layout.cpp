// d3d11-shader-object-layout.cpp
#include "d3d11-shader-object-layout.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

ShaderObjectLayoutImpl::SubObjectRangeOffset::SubObjectRangeOffset(
    slang::VariableLayoutReflection* varLayout)
    : BindingOffset(varLayout)
{
    if (auto pendingLayout = varLayout->getPendingDataLayout())
    {
        pendingOrdinaryData = (uint32_t)pendingLayout->getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM);
    }
}

ShaderObjectLayoutImpl::SubObjectRangeStride::SubObjectRangeStride(
    slang::TypeLayoutReflection* typeLayout)
    : BindingOffset(typeLayout)
{
    if (auto pendingLayout = typeLayout->getPendingDataTypeLayout())
    {
        pendingOrdinaryData = (uint32_t)typeLayout->getStride();
    }
}

Result ShaderObjectLayoutImpl::Builder::setElementTypeLayout(
    slang::TypeLayoutReflection* typeLayout)
{
    typeLayout = _unwrapParameterGroups(typeLayout, m_containerType);

    m_elementTypeLayout = typeLayout;

    m_totalOrdinaryDataSize = (uint32_t)typeLayout->getSize();

    // Compute the binding ranges that are used to store
    // the logical contents of the object in memory.

    SlangInt bindingRangeCount = typeLayout->getBindingRangeCount();
    for (SlangInt r = 0; r < bindingRangeCount; ++r)
    {
        slang::BindingType slangBindingType = typeLayout->getBindingRangeType(r);
        SlangInt count = typeLayout->getBindingRangeBindingCount(r);
        slang::TypeLayoutReflection* slangLeafTypeLayout =
            typeLayout->getBindingRangeLeafTypeLayout(r);

        BindingRangeInfo bindingRangeInfo;
        bindingRangeInfo.bindingType = slangBindingType;
        bindingRangeInfo.count = count;
        bindingRangeInfo.isSpecializable = typeLayout->isBindingRangeSpecializable(r);
        switch (slangBindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ExistentialValue:
            bindingRangeInfo.baseIndex = m_subObjectCount;
            bindingRangeInfo.subObjectIndex = m_subObjectCount;
            m_subObjectCount += count;
            break;
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
            if (slangLeafTypeLayout->getType()->getElementType() != nullptr)
            {
                // A structured buffer occupies both a resource slot and
                // a sub-object slot.
                bindingRangeInfo.subObjectIndex = m_subObjectCount;
                m_subObjectCount += count;
            }
            if (slangBindingType == slang::BindingType::RawBuffer)
            {
                bindingRangeInfo.baseIndex = m_srvCount;
                m_srvCount += count;
                m_srvRanges.add(r);
            }
            else
            {
                bindingRangeInfo.baseIndex = m_uavCount;
                m_uavCount += count;
                m_uavRanges.add(r);
            }
            break;
        case slang::BindingType::Sampler:
            bindingRangeInfo.baseIndex = m_samplerCount;
            m_samplerCount += count;
            m_samplerRanges.add(r);
            break;

        case slang::BindingType::CombinedTextureSampler:
            break;
        case slang::BindingType::MutableTexture:
        case slang::BindingType::MutableTypedBuffer:
            bindingRangeInfo.baseIndex = m_uavCount;
            m_uavCount += count;
            m_uavRanges.add(r);
            break;

        case slang::BindingType::VaryingInput:
            break;

        case slang::BindingType::VaryingOutput:
            break;

        default:
            bindingRangeInfo.baseIndex = m_srvCount;
            m_srvCount += count;
            m_srvRanges.add(r);
            break;
        }

        // We'd like to extract the information on the D3D11 shader
        // register that this range should bind into.
        //
        // A binding range represents a logical member of the shader
        // object type, and it may encompass zero or more *descriptor
        // ranges* that describe how it is physically bound to pipeline
        // state.
        //
        // If the current bindign range is backed by at least one descriptor
        // range then we can query the register offset of that descriptor
        // range. We expect that in the common case there will be exactly
        // one descriptor range, and we can extract the information easily.
        //
        // TODO: we might eventually need to special-case our handling
        // of combined texture-sampler ranges since they will need to
        // store two different offsets.
        //
        if (typeLayout->getBindingRangeDescriptorRangeCount(r) != 0)
        {
            // The Slang reflection information organizes the descriptor ranges
            // into "descriptor sets" but D3D11 has no notion like that so we
            // expect all ranges belong to a single set.
            //
            SlangInt descriptorSetIndex = typeLayout->getBindingRangeDescriptorSetIndex(r);
            SLANG_ASSERT(descriptorSetIndex == 0);

            SlangInt descriptorRangeIndex = typeLayout->getBindingRangeFirstDescriptorRangeIndex(r);
            auto registerOffset = typeLayout->getDescriptorSetDescriptorRangeIndexOffset(
                descriptorSetIndex,
                descriptorRangeIndex);

            bindingRangeInfo.registerOffset = (uint32_t)registerOffset;
        }

        m_bindingRanges.add(bindingRangeInfo);
    }

    SlangInt subObjectRangeCount = typeLayout->getSubObjectRangeCount();
    for (SlangInt r = 0; r < subObjectRangeCount; ++r)
    {
        SlangInt bindingRangeIndex = typeLayout->getSubObjectRangeBindingRangeIndex(r);
        auto& bindingRange = m_bindingRanges[bindingRangeIndex];

        auto slangBindingType = typeLayout->getBindingRangeType(bindingRangeIndex);
        slang::TypeLayoutReflection* slangLeafTypeLayout =
            typeLayout->getBindingRangeLeafTypeLayout(bindingRangeIndex);

        SubObjectRangeInfo subObjectRange;
        subObjectRange.bindingRangeIndex = bindingRangeIndex;

        // We will use Slang reflection information to extract the offset and stride
        // information for each sub-object range.
        //
        subObjectRange.offset = SubObjectRangeOffset(typeLayout->getSubObjectRangeOffset(r));
        subObjectRange.stride = SubObjectRangeStride(slangLeafTypeLayout);

        // A sub-object range can either represent a sub-object of a known
        // type, like a `ConstantBuffer<Foo>` or `ParameterBlock<Foo>`
        // *or* it can represent a sub-object of some existential type (e.g., `IBar`).
        //
        RefPtr<ShaderObjectLayoutImpl> subObjectLayout;
        switch (slangBindingType)
        {
        default:
            {
                // In the case of `ConstantBuffer<X>` or `ParameterBlock<X>`
                // we can construct a layout from the element type directly.
                //
                auto elementTypeLayout = slangLeafTypeLayout->getElementTypeLayout();
                createForElementType(
                    m_renderer,
                    m_session,
                    elementTypeLayout,
                    subObjectLayout.writeRef());
            }
            break;

        case slang::BindingType::ExistentialValue:
            // In the case of an interface-type sub-object range, we can only
            // construct a layout if we have static specialization information
            // that tells us what type we expect to find in that range.
            //
            // The static specialization information is expected to take the
            // form of a "pending" type layotu attached to the interface type
            // of the leaf type layout.
            //
            if (auto pendingTypeLayout = slangLeafTypeLayout->getPendingDataTypeLayout())
            {
                createForElementType(
                    m_renderer,
                    m_session,
                    pendingTypeLayout,
                    subObjectLayout.writeRef());

                // An interface-type range that includes ordinary data can
                // increase the size of the ordinary data buffer we need to
                // allocate for the parent object.
                //
                uint32_t ordinaryDataEnd =
                    subObjectRange.offset.pendingOrdinaryData +
                    (uint32_t)bindingRange.count * subObjectRange.stride.pendingOrdinaryData;

                if (ordinaryDataEnd > m_totalOrdinaryDataSize)
                {
                    m_totalOrdinaryDataSize = ordinaryDataEnd;
                }
            }
        }
        subObjectRange.layout = subObjectLayout;

        m_subObjectRanges.add(subObjectRange);
    }
    return SLANG_OK;
}

SlangResult ShaderObjectLayoutImpl::Builder::build(ShaderObjectLayoutImpl** outLayout)
{
    auto layout = RefPtr<ShaderObjectLayoutImpl>(new ShaderObjectLayoutImpl());
    SLANG_RETURN_ON_FAIL(layout->_init(this));

    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

Result ShaderObjectLayoutImpl::createForElementType(
    RendererBase* renderer,
    slang::ISession* session,
    slang::TypeLayoutReflection* elementType,
    ShaderObjectLayoutImpl** outLayout)
{
    Builder builder(renderer, session);
    builder.setElementTypeLayout(elementType);
    return builder.build(outLayout);
}

Result ShaderObjectLayoutImpl::_init(Builder const* builder)
{
    auto renderer = builder->m_renderer;

    initBase(renderer, builder->m_session, builder->m_elementTypeLayout);

    m_bindingRanges = builder->m_bindingRanges;
    m_srvRanges = builder->m_srvRanges;
    m_uavRanges = builder->m_uavRanges;
    m_samplerRanges = builder->m_samplerRanges;

    m_srvCount = builder->m_srvCount;
    m_samplerCount = builder->m_samplerCount;
    m_uavCount = builder->m_uavCount;
    m_subObjectCount = builder->m_subObjectCount;
    m_subObjectRanges = builder->m_subObjectRanges;

    m_totalOrdinaryDataSize = builder->m_totalOrdinaryDataSize;

    m_containerType = builder->m_containerType;
    return SLANG_OK;
}

Result RootShaderObjectLayoutImpl::Builder::build(RootShaderObjectLayoutImpl** outLayout)
{
    RefPtr<RootShaderObjectLayoutImpl> layout = new RootShaderObjectLayoutImpl();
    SLANG_RETURN_ON_FAIL(layout->_init(this));

    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

void RootShaderObjectLayoutImpl::Builder::addGlobalParams(
    slang::VariableLayoutReflection* globalsLayout)
{
    setElementTypeLayout(globalsLayout->getTypeLayout());
    m_pendingDataOffset = BindingOffset(globalsLayout).pending;
}

void RootShaderObjectLayoutImpl::Builder::addEntryPoint(
    SlangStage stage,
    ShaderObjectLayoutImpl* entryPointLayout,
    slang::EntryPointLayout* slangEntryPoint)
{
    EntryPointInfo info;
    info.layout = entryPointLayout;
    info.offset = BindingOffset(slangEntryPoint->getVarLayout());
    m_entryPoints.add(info);
}

Result RootShaderObjectLayoutImpl::create(
    RendererBase* renderer,
    slang::IComponentType* program,
    slang::ProgramLayout* programLayout,
    RootShaderObjectLayoutImpl** outLayout)
{
    RootShaderObjectLayoutImpl::Builder builder(renderer, program, programLayout);
    builder.addGlobalParams(programLayout->getGlobalParamsVarLayout());

    SlangInt entryPointCount = programLayout->getEntryPointCount();
    for (SlangInt e = 0; e < entryPointCount; ++e)
    {
        auto slangEntryPoint = programLayout->getEntryPointByIndex(e);
        RefPtr<ShaderObjectLayoutImpl> entryPointLayout;
        SLANG_RETURN_ON_FAIL(ShaderObjectLayoutImpl::createForElementType(
            renderer,
            program->getSession(),
            slangEntryPoint->getTypeLayout(),
            entryPointLayout.writeRef()));
        builder.addEntryPoint(slangEntryPoint->getStage(), entryPointLayout, slangEntryPoint);
    }

    SLANG_RETURN_ON_FAIL(builder.build(outLayout));

    return SLANG_OK;
}

Result RootShaderObjectLayoutImpl::_init(Builder const* builder)
{
    auto renderer = builder->m_renderer;

    SLANG_RETURN_ON_FAIL(Super::_init(builder));

    m_program = builder->m_program;
    m_programLayout = builder->m_programLayout;
    m_entryPoints = builder->m_entryPoints;
    m_pendingDataOffset = builder->m_pendingDataOffset;
    m_slangSession = m_program->getSession();

    return SLANG_OK;
}

} // namespace d3d11
} // namespace gfx
