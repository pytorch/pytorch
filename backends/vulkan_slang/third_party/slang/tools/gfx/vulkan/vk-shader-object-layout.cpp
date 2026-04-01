// vk-shader-object-layout.cpp
#include "vk-shader-object-layout.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

Index ShaderObjectLayoutImpl::Builder::findOrAddDescriptorSet(Index space)
{
    Index index;
    if (m_mapSpaceToDescriptorSetIndex.tryGetValue(space, index))
        return index;

    DescriptorSetInfo info = {};
    info.space = space;

    index = m_descriptorSetBuildInfos.getCount();
    m_descriptorSetBuildInfos.add(info);

    m_mapSpaceToDescriptorSetIndex.add(space, index);
    return index;
}

VkDescriptorType ShaderObjectLayoutImpl::Builder::_mapDescriptorType(
    slang::BindingType slangBindingType)
{
    switch (slangBindingType)
    {
    case slang::BindingType::PushConstant:
    default:
        SLANG_ASSERT("unsupported binding type");
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;

    case slang::BindingType::Sampler:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    case slang::BindingType::CombinedTextureSampler:
        return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case slang::BindingType::Texture:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case slang::BindingType::MutableTexture:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case slang::BindingType::TypedBuffer:
        return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case slang::BindingType::MutableTypedBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    case slang::BindingType::RawBuffer:
    case slang::BindingType::MutableRawBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case slang::BindingType::InputRenderTarget:
        return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    case slang::BindingType::InlineUniformData:
        return VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT;
    case slang::BindingType::RayTracingAccelerationStructure:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    case slang::BindingType::ConstantBuffer:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }
}

/// Add any descriptor ranges implied by this object containing a leaf
/// sub-object described by `typeLayout`, at the given `offset`.

void ShaderObjectLayoutImpl::Builder::_addDescriptorRangesAsValue(
    slang::TypeLayoutReflection* typeLayout,
    BindingOffset const& offset)
{
    // First we will scan through all the descriptor sets that the Slang reflection
    // information believes go into making up the given type.
    //
    // Note: We are initializing the sets in order so that their order in our
    // internal data structures should be deterministically based on the order
    // in which they are listed in Slang's reflection information.
    //
    Index descriptorSetCount = typeLayout->getDescriptorSetCount();
    for (Index i = 0; i < descriptorSetCount; ++i)
    {
        SlangInt descriptorRangeCount = typeLayout->getDescriptorSetDescriptorRangeCount(i);
        if (descriptorRangeCount == 0)
            continue;
        auto descriptorSetIndex =
            findOrAddDescriptorSet(offset.bindingSet + typeLayout->getDescriptorSetSpaceOffset(i));
    }

    // For actually populating the descriptor sets we prefer to enumerate
    // the binding ranges of the type instead of the descriptor sets.
    //
    Index bindRangeCount = typeLayout->getBindingRangeCount();
    for (Index i = 0; i < bindRangeCount; ++i)
    {
        auto bindingRangeIndex = i;
        auto bindingRangeType = typeLayout->getBindingRangeType(bindingRangeIndex);
        switch (bindingRangeType)
        {
        default:
            break;

        // We will skip over ranges that represent sub-objects for now, and handle
        // them in a separate pass.
        //
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ExistentialValue:
        case slang::BindingType::PushConstant:
            continue;
        }

        // Given a binding range we are interested in, we will then enumerate
        // its contained descriptor ranges.

        Index descriptorRangeCount =
            typeLayout->getBindingRangeDescriptorRangeCount(bindingRangeIndex);
        if (descriptorRangeCount == 0)
            continue;
        auto slangDescriptorSetIndex =
            typeLayout->getBindingRangeDescriptorSetIndex(bindingRangeIndex);
        auto descriptorSetIndex = findOrAddDescriptorSet(
            offset.bindingSet + typeLayout->getDescriptorSetSpaceOffset(slangDescriptorSetIndex));
        auto& descriptorSetInfo = m_descriptorSetBuildInfos[descriptorSetIndex];

        Index firstDescriptorRangeIndex =
            typeLayout->getBindingRangeFirstDescriptorRangeIndex(bindingRangeIndex);
        for (Index j = 0; j < descriptorRangeCount; ++j)
        {
            Index descriptorRangeIndex = firstDescriptorRangeIndex + j;
            auto slangDescriptorType = typeLayout->getDescriptorSetDescriptorRangeType(
                slangDescriptorSetIndex,
                descriptorRangeIndex);

            // Certain kinds of descriptor ranges reflected by Slang do not
            // manifest as descriptors at the Vulkan level, so we will skip those.
            //
            switch (slangDescriptorType)
            {
            case slang::BindingType::ExistentialValue:
            case slang::BindingType::InlineUniformData:
            case slang::BindingType::PushConstant:
                continue;
            default:
                break;
            }

            auto vkDescriptorType = _mapDescriptorType(slangDescriptorType);
            VkDescriptorSetLayoutBinding vkBindingRangeDesc = {};
            vkBindingRangeDesc.binding =
                offset.binding + (uint32_t)typeLayout->getDescriptorSetDescriptorRangeIndexOffset(
                                     slangDescriptorSetIndex,
                                     descriptorRangeIndex);
            vkBindingRangeDesc.descriptorCount =
                (uint32_t)typeLayout->getDescriptorSetDescriptorRangeDescriptorCount(
                    slangDescriptorSetIndex,
                    descriptorRangeIndex);
            vkBindingRangeDesc.descriptorType = vkDescriptorType;
            vkBindingRangeDesc.stageFlags = VK_SHADER_STAGE_ALL;

            descriptorSetInfo.vkBindings.add(vkBindingRangeDesc);
        }
    }

    // We skipped over the sub-object ranges when adding descriptors above,
    // and now we will address that oversight by iterating over just
    // the sub-object ranges.
    //
    Index subObjectRangeCount = typeLayout->getSubObjectRangeCount();
    for (Index subObjectRangeIndex = 0; subObjectRangeIndex < subObjectRangeCount;
         ++subObjectRangeIndex)
    {
        auto bindingRangeIndex =
            typeLayout->getSubObjectRangeBindingRangeIndex(subObjectRangeIndex);
        auto bindingType = typeLayout->getBindingRangeType(bindingRangeIndex);

        auto subObjectTypeLayout = typeLayout->getBindingRangeLeafTypeLayout(bindingRangeIndex);
        SLANG_ASSERT(subObjectTypeLayout);

        BindingOffset subObjectRangeOffset = offset;
        subObjectRangeOffset +=
            BindingOffset(typeLayout->getSubObjectRangeOffset(subObjectRangeIndex));

        switch (bindingType)
        {
        // A `ParameterBlock<X>` never contributes descripto ranges to the
        // decriptor sets of a parent object.
        //
        case slang::BindingType::ParameterBlock:
        default:
            break;

        case slang::BindingType::ExistentialValue:
            // An interest/existential-typed sub-object range will only contribute
            // descriptor ranges to a parent object in the case where it has been
            // specialied, which is precisely the case where the Slang reflection
            // information will tell us about its "pending" layout.
            //
            if (auto pendingTypeLayout = subObjectTypeLayout->getPendingDataTypeLayout())
            {
                BindingOffset pendingOffset = BindingOffset(subObjectRangeOffset.pending);
                _addDescriptorRangesAsValue(pendingTypeLayout, pendingOffset);
            }
            break;

        case slang::BindingType::ConstantBuffer:
            {
                // A `ConstantBuffer<X>` range will contribute any nested descriptor
                // ranges in `X`, along with a leading descriptor range for a
                // uniform buffer to hold ordinary/uniform data, if there is any.

                SLANG_ASSERT(subObjectTypeLayout);

                auto containerVarLayout = subObjectTypeLayout->getContainerVarLayout();
                SLANG_ASSERT(containerVarLayout);

                auto elementVarLayout = subObjectTypeLayout->getElementVarLayout();
                SLANG_ASSERT(elementVarLayout);

                auto elementTypeLayout = elementVarLayout->getTypeLayout();
                SLANG_ASSERT(elementTypeLayout);

                BindingOffset containerOffset = subObjectRangeOffset;
                containerOffset += BindingOffset(subObjectTypeLayout->getContainerVarLayout());

                BindingOffset elementOffset = subObjectRangeOffset;
                elementOffset += BindingOffset(elementVarLayout);

                _addDescriptorRangesAsConstantBuffer(
                    elementTypeLayout,
                    containerOffset,
                    elementOffset);
            }
            break;

        case slang::BindingType::PushConstant:
            {
                // This case indicates a `ConstantBuffer<X>` that was marked as being
                // used for push constants.
                //
                // Much of the handling is the same as for an ordinary
                // `ConstantBuffer<X>`, but of course we need to handle the ordinary
                // data part differently.

                SLANG_ASSERT(subObjectTypeLayout);

                auto containerVarLayout = subObjectTypeLayout->getContainerVarLayout();
                SLANG_ASSERT(containerVarLayout);

                auto elementVarLayout = subObjectTypeLayout->getElementVarLayout();
                SLANG_ASSERT(elementVarLayout);

                auto elementTypeLayout = elementVarLayout->getTypeLayout();
                SLANG_ASSERT(elementTypeLayout);

                BindingOffset containerOffset = subObjectRangeOffset;
                containerOffset += BindingOffset(subObjectTypeLayout->getContainerVarLayout());

                BindingOffset elementOffset = subObjectRangeOffset;
                elementOffset += BindingOffset(elementVarLayout);

                _addDescriptorRangesAsPushConstantBuffer(
                    elementTypeLayout,
                    containerOffset,
                    elementOffset);
            }
            break;
        }
    }
}

/// Add the descriptor ranges implied by a `ConstantBuffer<X>` where `X` is
/// described by `elementTypeLayout`.
///
/// The `containerOffset` and `elementOffset` are the binding offsets that
/// should apply to the buffer itself and the contents of the buffer, respectively.
///

void ShaderObjectLayoutImpl::Builder::_addDescriptorRangesAsConstantBuffer(
    slang::TypeLayoutReflection* elementTypeLayout,
    BindingOffset const& containerOffset,
    BindingOffset const& elementOffset)
{
    // If the type has ordinary uniform data fields, we need to make sure to create
    // a descriptor set with a constant buffer binding in the case that the shader
    // object is bound as a stand alone parameter block.
    if (elementTypeLayout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM) != 0)
    {
        auto descriptorSetIndex = findOrAddDescriptorSet(containerOffset.bindingSet);
        auto& descriptorSetInfo = m_descriptorSetBuildInfos[descriptorSetIndex];
        VkDescriptorSetLayoutBinding vkBindingRangeDesc = {};
        vkBindingRangeDesc.binding = containerOffset.binding;
        vkBindingRangeDesc.descriptorCount = 1;
        vkBindingRangeDesc.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        vkBindingRangeDesc.stageFlags = VK_SHADER_STAGE_ALL;
        descriptorSetInfo.vkBindings.add(vkBindingRangeDesc);
    }

    _addDescriptorRangesAsValue(elementTypeLayout, elementOffset);
}

/// Add the descriptor ranges implied by a `PushConstantBuffer<X>` where `X` is
/// described by `elementTypeLayout`.
///
/// The `containerOffset` and `elementOffset` are the binding offsets that
/// should apply to the buffer itself and the contents of the buffer, respectively.
///

void ShaderObjectLayoutImpl::Builder::_addDescriptorRangesAsPushConstantBuffer(
    slang::TypeLayoutReflection* elementTypeLayout,
    BindingOffset const& containerOffset,
    BindingOffset const& elementOffset)
{
    // If the type has ordinary uniform data fields, we need to make sure to create
    // a descriptor set with a constant buffer binding in the case that the shader
    // object is bound as a stand alone parameter block.
    auto ordinaryDataSize = (uint32_t)elementTypeLayout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
    if (ordinaryDataSize != 0)
    {
        auto pushConstantRangeIndex = containerOffset.pushConstantRange;

        VkPushConstantRange vkPushConstantRange = {};
        vkPushConstantRange.size = ordinaryDataSize;
        vkPushConstantRange.stageFlags = VK_SHADER_STAGE_ALL; // TODO: be more clever

        while ((uint32_t)m_ownPushConstantRanges.getCount() <= pushConstantRangeIndex)
        {
            VkPushConstantRange emptyRange = {0};
            m_ownPushConstantRanges.add(emptyRange);
        }

        m_ownPushConstantRanges[pushConstantRangeIndex] = vkPushConstantRange;
    }

    _addDescriptorRangesAsValue(elementTypeLayout, elementOffset);
}

/// Add binding ranges to this shader object layout, as implied by the given
/// `typeLayout`

void ShaderObjectLayoutImpl::Builder::addBindingRanges(slang::TypeLayoutReflection* typeLayout)
{
    SlangInt bindingRangeCount = typeLayout->getBindingRangeCount();
    for (SlangInt r = 0; r < bindingRangeCount; ++r)
    {
        slang::BindingType slangBindingType = typeLayout->getBindingRangeType(r);
        uint32_t count = (uint32_t)typeLayout->getBindingRangeBindingCount(r);
        slang::TypeLayoutReflection* slangLeafTypeLayout =
            typeLayout->getBindingRangeLeafTypeLayout(r);

        Index baseIndex = 0;
        Index subObjectIndex = 0;
        switch (slangBindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ExistentialValue:
            baseIndex = m_subObjectCount;
            subObjectIndex = baseIndex;
            m_subObjectCount += count;
            break;
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
            if (slangLeafTypeLayout->getType()->getElementType() != nullptr)
            {
                // A structured buffer occupies both a resource slot and
                // a sub-object slot.
                subObjectIndex = m_subObjectCount;
                m_subObjectCount += count;
            }
            baseIndex = m_resourceViewCount;
            m_resourceViewCount += count;
            break;
        case slang::BindingType::Sampler:
            baseIndex = m_samplerCount;
            m_samplerCount += count;
            m_totalBindingCount += 1;
            break;

        case slang::BindingType::CombinedTextureSampler:
            baseIndex = m_combinedTextureSamplerCount;
            m_combinedTextureSamplerCount += count;
            m_totalBindingCount += 1;
            break;

        case slang::BindingType::VaryingInput:
            baseIndex = m_varyingInputCount;
            m_varyingInputCount += count;
            break;

        case slang::BindingType::VaryingOutput:
            baseIndex = m_varyingOutputCount;
            m_varyingOutputCount += count;
            break;
        default:
            baseIndex = m_resourceViewCount;
            m_resourceViewCount += count;
            m_totalBindingCount += 1;
            break;
        }

        BindingRangeInfo bindingRangeInfo;
        bindingRangeInfo.bindingType = slangBindingType;
        bindingRangeInfo.count = count;
        bindingRangeInfo.baseIndex = baseIndex;
        bindingRangeInfo.subObjectIndex = subObjectIndex;
        bindingRangeInfo.isSpecializable = typeLayout->isBindingRangeSpecializable(r);
        // We'd like to extract the information on the GLSL/SPIR-V
        // `binding` that this range should bind into (or whatever
        // other specific kind of offset/index is appropriate to it).
        //
        // A binding range represents a logical member of the shader
        // object type, and it may encompass zero or more *descriptor
        // ranges* that describe how it is physically bound to pipeline
        // state.
        //
        // If the current bindign range is backed by at least one descriptor
        // range then we can query the binding offset of that descriptor
        // range. We expect that in the common case there will be exactly
        // one descriptor range, and we can extract the information easily.
        //
        if (typeLayout->getBindingRangeDescriptorRangeCount(r) != 0)
        {
            SlangInt descriptorSetIndex = typeLayout->getBindingRangeDescriptorSetIndex(r);
            SlangInt descriptorRangeIndex = typeLayout->getBindingRangeFirstDescriptorRangeIndex(r);

            auto set = typeLayout->getDescriptorSetSpaceOffset(descriptorSetIndex);
            auto bindingOffset = typeLayout->getDescriptorSetDescriptorRangeIndexOffset(
                descriptorSetIndex,
                descriptorRangeIndex);

            bindingRangeInfo.setOffset = uint32_t(set);
            bindingRangeInfo.bindingOffset = uint32_t(bindingOffset);
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

        // A sub-object range can either represent a sub-object of a known
        // type, like a `ConstantBuffer<Foo>` or `ParameterBlock<Foo>`
        // (in which case we can pre-compute a layout to use, based on
        // the type `Foo`) *or* it can represent a sub-object of some
        // existential type (e.g., `IBar`) in which case we cannot
        // know the appropraite type/layout of sub-object to allocate.
        //
        RefPtr<ShaderObjectLayoutImpl> subObjectLayout;
        switch (slangBindingType)
        {
        default:
            {
                auto varLayout = slangLeafTypeLayout->getElementVarLayout();
                auto subTypeLayout = varLayout->getTypeLayout();
                ShaderObjectLayoutImpl::createForElementType(
                    m_renderer,
                    m_session,
                    subTypeLayout,
                    subObjectLayout.writeRef());
            }
            break;

        case slang::BindingType::ExistentialValue:
            if (auto pendingTypeLayout = slangLeafTypeLayout->getPendingDataTypeLayout())
            {
                ShaderObjectLayoutImpl::createForElementType(
                    m_renderer,
                    m_session,
                    pendingTypeLayout,
                    subObjectLayout.writeRef());
            }
            break;
        }

        SubObjectRangeInfo subObjectRange;
        subObjectRange.bindingRangeIndex = bindingRangeIndex;
        subObjectRange.layout = subObjectLayout;

        // We will use Slang reflection infromation to extract the offset information
        // for each sub-object range.
        //
        // TODO: We should also be extracting the uniform offset here.
        //
        subObjectRange.offset = SubObjectRangeOffset(typeLayout->getSubObjectRangeOffset(r));
        subObjectRange.stride = SubObjectRangeStride(slangLeafTypeLayout);

        switch (slangBindingType)
        {
        case slang::BindingType::ParameterBlock:
            m_childDescriptorSetCount += subObjectLayout->getTotalDescriptorSetCount();
            m_childPushConstantRangeCount += subObjectLayout->getTotalPushConstantRangeCount();
            break;

        case slang::BindingType::ConstantBuffer:
            m_childDescriptorSetCount += subObjectLayout->getChildDescriptorSetCount();
            m_totalBindingCount += subObjectLayout->getTotalBindingCount();
            m_childPushConstantRangeCount += subObjectLayout->getTotalPushConstantRangeCount();
            break;

        case slang::BindingType::ExistentialValue:
            if (subObjectLayout)
            {
                m_childDescriptorSetCount += subObjectLayout->getChildDescriptorSetCount();
                m_totalBindingCount += subObjectLayout->getTotalBindingCount();
                m_childPushConstantRangeCount += subObjectLayout->getTotalPushConstantRangeCount();

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
            break;

        default:
            break;
        }

        m_subObjectRanges.add(subObjectRange);
    }
}

Result ShaderObjectLayoutImpl::Builder::setElementTypeLayout(
    slang::TypeLayoutReflection* typeLayout)
{
    typeLayout = _unwrapParameterGroups(typeLayout, m_containerType);
    m_elementTypeLayout = typeLayout;

    m_totalOrdinaryDataSize = (uint32_t)typeLayout->getSize();

    // Next we will compute the binding ranges that are used to store
    // the logical contents of the object in memory. These will relate
    // to the descriptor ranges in the various sets, but not always
    // in a one-to-one fashion.

    addBindingRanges(typeLayout);

    // Note: This routine does not take responsibility for
    // adding descriptor ranges at all, because the exact way
    // that descriptor ranges need to be added varies between
    // ordinary shader objects, root shader objects, and entry points.

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
    DeviceImpl* renderer,
    slang::ISession* session,
    slang::TypeLayoutReflection* elementType,
    ShaderObjectLayoutImpl** outLayout)
{
    Builder builder(renderer, session);
    builder.setElementTypeLayout(elementType);

    // When constructing a shader object layout directly from a reflected
    // type in Slang, we want to compute the descriptor sets and ranges
    // that would be used if this object were bound as a parameter block.
    //
    // It might seem like we need to deal with the other cases for how
    // the shader object might be bound, but the descriptor ranges we
    // compute here will only ever be used in parameter-block case.
    //
    // One important wrinkle is that we know that the parameter block
    // allocated for `elementType` will potentially need a buffer `binding`
    // for any ordinary data it contains.

    bool needsOrdinaryDataBuffer =
        builder.m_elementTypeLayout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM) != 0;
    uint32_t ordinaryDataBufferCount = needsOrdinaryDataBuffer ? 1 : 0;

    // When binding the object, we know that the ordinary data buffer will
    // always use a the first available `binding`, so its offset will be
    // all zeroes.
    //
    BindingOffset containerOffset;

    // In contrast, the `binding`s used by all the other entries in the
    // parameter block will need to be offset by one if there was
    // an ordinary data buffer.
    //
    BindingOffset elementOffset;
    elementOffset.binding = ordinaryDataBufferCount;

    // Furthermore, any `binding`s that arise due to "pending" data
    // in the type of the object (due to specialization for existential types)
    // will need to come after all the other `binding`s that were
    // part of the "primary" (unspecialized) data.
    //
    uint32_t primaryDescriptorCount =
        ordinaryDataBufferCount + (uint32_t)builder.m_elementTypeLayout->getSize(
                                      SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT);
    elementOffset.pending.binding = primaryDescriptorCount;

    // Once we've computed the offset information, we simply add the
    // descriptor ranges as if things were declared as a `ConstantBuffer<X>`,
    // since that is how things will be laid out inside the parameter block.
    //
    builder._addDescriptorRangesAsConstantBuffer(
        builder.m_elementTypeLayout,
        containerOffset,
        elementOffset);
    return builder.build(outLayout);
}

ShaderObjectLayoutImpl::~ShaderObjectLayoutImpl()
{
    for (auto& descSetInfo : m_descriptorSetInfos)
    {
        getDevice()->m_api.vkDestroyDescriptorSetLayout(
            getDevice()->m_api.m_device,
            descSetInfo.descriptorSetLayout,
            nullptr);
    }
}

Result ShaderObjectLayoutImpl::_init(Builder const* builder)
{
    auto renderer = builder->m_renderer;

    initBase(renderer, builder->m_session, builder->m_elementTypeLayout);

    m_bindingRanges = builder->m_bindingRanges;

    m_descriptorSetInfos = _Move(builder->m_descriptorSetBuildInfos);
    m_ownPushConstantRanges = builder->m_ownPushConstantRanges;
    m_resourceViewCount = builder->m_resourceViewCount;
    m_samplerCount = builder->m_samplerCount;
    m_combinedTextureSamplerCount = builder->m_combinedTextureSamplerCount;
    m_childDescriptorSetCount = builder->m_childDescriptorSetCount;
    m_totalBindingCount = builder->m_totalBindingCount;
    m_subObjectCount = builder->m_subObjectCount;
    m_subObjectRanges = builder->m_subObjectRanges;
    m_totalOrdinaryDataSize = builder->m_totalOrdinaryDataSize;

    m_containerType = builder->m_containerType;

    // Create VkDescriptorSetLayout for all descriptor sets.
    for (auto& descriptorSetInfo : m_descriptorSetInfos)
    {
        VkDescriptorSetLayoutCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.pBindings = descriptorSetInfo.vkBindings.getBuffer();
        createInfo.bindingCount = (uint32_t)descriptorSetInfo.vkBindings.getCount();
        VkDescriptorSetLayout vkDescSetLayout;
        SLANG_RETURN_ON_FAIL(renderer->m_api.vkCreateDescriptorSetLayout(
            renderer->m_api.m_device,
            &createInfo,
            nullptr,
            &vkDescSetLayout));
        descriptorSetInfo.descriptorSetLayout = vkDescSetLayout;
    }
    return SLANG_OK;
}

DeviceImpl* ShaderObjectLayoutImpl::getDevice()
{
    return static_cast<DeviceImpl*>(m_renderer);
}

Result EntryPointLayout::Builder::build(EntryPointLayout** outLayout)
{
    RefPtr<EntryPointLayout> layout = new EntryPointLayout();
    SLANG_RETURN_ON_FAIL(layout->_init(this));

    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

void EntryPointLayout::Builder::addEntryPointParams(slang::EntryPointLayout* entryPointLayout)
{
    m_slangEntryPointLayout = entryPointLayout;
    setElementTypeLayout(entryPointLayout->getTypeLayout());
    m_shaderStageFlag = VulkanUtil::getShaderStage(entryPointLayout->getStage());

    // Note: we do not bother adding any descriptor sets/ranges here,
    // because the descriptor ranges of an entry point will simply
    // be allocated as part of the descriptor sets for the root
    // shader object.
}

Result EntryPointLayout::_init(Builder const* builder)
{
    auto renderer = builder->m_renderer;

    SLANG_RETURN_ON_FAIL(Super::_init(builder));

    m_slangEntryPointLayout = builder->m_slangEntryPointLayout;
    m_shaderStageFlag = builder->m_shaderStageFlag;
    return SLANG_OK;
}

RootShaderObjectLayout::~RootShaderObjectLayout()
{
    if (m_pipelineLayout)
    {
        m_renderer->m_api.vkDestroyPipelineLayout(
            m_renderer->m_api.m_device,
            m_pipelineLayout,
            nullptr);
    }
}

Index RootShaderObjectLayout::findEntryPointIndex(VkShaderStageFlags stage)
{
    auto entryPointCount = m_entryPoints.getCount();
    for (Index i = 0; i < entryPointCount; ++i)
    {
        auto entryPoint = m_entryPoints[i];
        if (entryPoint.layout->getShaderStageFlag() == stage)
            return i;
    }
    return -1;
}

Result RootShaderObjectLayout::create(
    DeviceImpl* renderer,
    slang::IComponentType* program,
    slang::ProgramLayout* programLayout,
    RootShaderObjectLayout** outLayout)
{
    RootShaderObjectLayout::Builder builder(renderer, program, programLayout);
    builder.addGlobalParams(programLayout->getGlobalParamsVarLayout());

    SlangInt entryPointCount = programLayout->getEntryPointCount();
    for (SlangInt e = 0; e < entryPointCount; ++e)
    {
        auto slangEntryPoint = programLayout->getEntryPointByIndex(e);

        EntryPointLayout::Builder entryPointBuilder(renderer, program->getSession());
        entryPointBuilder.addEntryPointParams(slangEntryPoint);

        RefPtr<EntryPointLayout> entryPointLayout;
        SLANG_RETURN_ON_FAIL(entryPointBuilder.build(entryPointLayout.writeRef()));

        builder.addEntryPoint(entryPointLayout);
    }

    SLANG_RETURN_ON_FAIL(builder.build(outLayout));

    return SLANG_OK;
}

Result RootShaderObjectLayout::_init(Builder const* builder)
{
    auto renderer = builder->m_renderer;

    SLANG_RETURN_ON_FAIL(Super::_init(builder));

    m_program = builder->m_program;
    m_programLayout = builder->m_programLayout;
    m_entryPoints = _Move(builder->m_entryPoints);
    m_pendingDataOffset = builder->m_pendingDataOffset;
    m_renderer = renderer;

    // If the program has unbound specialization parameters,
    // then we will avoid creating a final Vulkan pipeline layout.
    //
    // TODO: We should really create the information necessary
    // for binding as part of a separate object, so that we have
    // a clean seperation between what is needed for writing into
    // a shader object vs. what is needed for binding it to the
    // pipeline. We eventually need to be able to create bindable
    // state objects from unspecialized programs, in order to
    // support dynamic dispatch.
    //
    if (m_program->getSpecializationParamCount() != 0)
        return SLANG_OK;

    // Otherwise, we need to create a final (bindable) layout.
    //
    // We will use a recursive walk to collect all the `VkDescriptorSetLayout`s
    // that are required for the global scope, sub-objects, and entry points.
    //
    SLANG_RETURN_ON_FAIL(addAllDescriptorSets());

    // We will also use a recursive walk to collect all the push-constant
    // ranges needed for this object, sub-objects, and entry points.
    //
    SLANG_RETURN_ON_FAIL(addAllPushConstantRanges());

    // Once we've collected the information across the entire
    // tree of sub-objects

    // Now call Vulkan API to create a pipeline layout.
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = (uint32_t)m_vkDescriptorSetLayouts.getCount();
    pipelineLayoutCreateInfo.pSetLayouts = m_vkDescriptorSetLayouts.getBuffer();
    if (m_allPushConstantRanges.getCount())
    {
        pipelineLayoutCreateInfo.pushConstantRangeCount =
            (uint32_t)m_allPushConstantRanges.getCount();
        pipelineLayoutCreateInfo.pPushConstantRanges = m_allPushConstantRanges.getBuffer();
    }
    SLANG_RETURN_ON_FAIL(m_renderer->m_api.vkCreatePipelineLayout(
        m_renderer->m_api.m_device,
        &pipelineLayoutCreateInfo,
        nullptr,
        &m_pipelineLayout));
    return SLANG_OK;
}

/// Add all the descriptor sets implied by this root object and sub-objects

Result RootShaderObjectLayout::addAllDescriptorSets()
{
    SLANG_RETURN_ON_FAIL(addAllDescriptorSetsRec(this));

    // Note: the descriptor ranges/sets for direct entry point parameters
    // were already enumerated into the ranges/sets of the root object itself,
    // so we don't wnat to add them again.
    //
    // We do however have to deal with the possibility that an entry
    // point could introduce "child" descriptor sets, e.g., because it
    // has a `ParameterBlock<X>` parameter.
    //
    for (auto& entryPoint : getEntryPoints())
    {
        SLANG_RETURN_ON_FAIL(addChildDescriptorSetsRec(entryPoint.layout));
    }

    return SLANG_OK;
}

/// Recurisvely add descriptor sets defined by `layout` and sub-objects

Result RootShaderObjectLayout::addAllDescriptorSetsRec(ShaderObjectLayoutImpl* layout)
{
    // TODO: This logic assumes that descriptor sets are all contiguous
    // and have been allocated in a global order that matches the order
    // of enumeration here.

    for (auto& descSetInfo : layout->getOwnDescriptorSets())
    {
        m_vkDescriptorSetLayouts.add(descSetInfo.descriptorSetLayout);
    }

    SLANG_RETURN_ON_FAIL(addChildDescriptorSetsRec(layout));
    return SLANG_OK;
}

/// Recurisvely add descriptor sets defined by sub-objects of `layout`

Result RootShaderObjectLayout::addChildDescriptorSetsRec(ShaderObjectLayoutImpl* layout)
{
    for (auto& subObject : layout->getSubObjectRanges())
    {
        auto bindingRange = layout->getBindingRange(subObject.bindingRangeIndex);
        switch (bindingRange.bindingType)
        {
        case slang::BindingType::ParameterBlock:
            SLANG_RETURN_ON_FAIL(addAllDescriptorSetsRec(subObject.layout));
            break;

        default:
            if (auto subObjectLayout = subObject.layout)
            {
                SLANG_RETURN_ON_FAIL(addChildDescriptorSetsRec(subObject.layout));
            }
            break;
        }
    }

    return SLANG_OK;
}

/// Add all the push-constant ranges implied by this root object and sub-objects

Result RootShaderObjectLayout::addAllPushConstantRanges()
{
    SLANG_RETURN_ON_FAIL(addAllPushConstantRangesRec(this));

    for (auto& entryPoint : getEntryPoints())
    {
        SLANG_RETURN_ON_FAIL(addChildPushConstantRangesRec(entryPoint.layout));
    }

    return SLANG_OK;
}

/// Recurisvely add push-constant ranges defined by `layout` and sub-objects

Result RootShaderObjectLayout::addAllPushConstantRangesRec(ShaderObjectLayoutImpl* layout)
{
    // TODO: This logic assumes that push-constant ranges are all contiguous
    // and have been allocated in a global order that matches the order
    // of enumeration here.

    for (auto pushConstantRange : layout->getOwnPushConstantRanges())
    {
        pushConstantRange.offset = m_totalPushConstantSize;
        m_totalPushConstantSize += pushConstantRange.size;

        m_allPushConstantRanges.add(pushConstantRange);
    }

    SLANG_RETURN_ON_FAIL(addChildPushConstantRangesRec(layout));
    return SLANG_OK;
}

/// Recurisvely add push-constant ranges defined by sub-objects of `layout`

Result RootShaderObjectLayout::addChildPushConstantRangesRec(ShaderObjectLayoutImpl* layout)
{
    for (auto& subObject : layout->getSubObjectRanges())
    {
        if (auto subObjectLayout = subObject.layout)
        {
            SLANG_RETURN_ON_FAIL(addAllPushConstantRangesRec(subObject.layout));
        }
    }

    return SLANG_OK;
}

Result RootShaderObjectLayout::Builder::build(RootShaderObjectLayout** outLayout)
{
    RefPtr<RootShaderObjectLayout> layout = new RootShaderObjectLayout();
    SLANG_RETURN_ON_FAIL(layout->_init(this));
    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

void RootShaderObjectLayout::Builder::addGlobalParams(
    slang::VariableLayoutReflection* globalsLayout)
{
    setElementTypeLayout(globalsLayout->getTypeLayout());

    // We need to populate our descriptor sets/ranges with information
    // from the layout of the global scope.
    //
    // While we expect that the parameter in the global scope start
    // at an offset of zero, it is also worth querying the offset
    // information because it could impact the locations assigned
    // to "pending" data in the case of static specialization.
    //
    BindingOffset offset(globalsLayout);

    // Note: We are adding descriptor ranges here based directly on
    // the type of the global-scope layout. The type layout for the
    // global scope will either be something like a `struct GlobalParams`
    // that contains all the global-scope parameters or a `ConstantBuffer<GlobalParams>`
    // and in either case the `_addDescriptorRangesAsValue` can properly
    // add all the ranges implied.
    //
    // As a result we don't require any special-case logic here to
    // deal with the possibility of a "default" constant buffer allocated
    // for global-scope parameters of uniform/ordinary type.
    //
    _addDescriptorRangesAsValue(globalsLayout->getTypeLayout(), offset);

    // We want to keep track of the offset that was applied to "pending"
    // data because we will need it again later when it comes time to
    // actually bind things.
    //
    m_pendingDataOffset = offset.pending;
}

void RootShaderObjectLayout::Builder::addEntryPoint(EntryPointLayout* entryPointLayout)
{
    auto slangEntryPointLayout = entryPointLayout->getSlangLayout();
    auto entryPointVarLayout = slangEntryPointLayout->getVarLayout();

    // The offset information for each entry point needs to
    // be adjusted by any offset for "pending" data that
    // was recorded in the global-scope layout.
    //
    // TODO(tfoley): Double-check that this is correct.

    BindingOffset entryPointOffset(entryPointVarLayout);
    entryPointOffset.pending += m_pendingDataOffset;

    EntryPointInfo info;
    info.layout = entryPointLayout;
    info.offset = entryPointOffset;

    // Similar to the case for the global scope, we expect the
    // type layout for the entry point parameters to be either
    // a `struct EntryPointParams` or a `PushConstantBuffer<EntryPointParams>`.
    // Rather than deal with the different cases here, we will
    // trust the `_addDescriptorRangesAsValue` code to handle
    // either case correctly.
    //
    _addDescriptorRangesAsValue(entryPointVarLayout->getTypeLayout(), entryPointOffset);

    m_entryPoints.add(info);
}

} // namespace vk
} // namespace gfx
