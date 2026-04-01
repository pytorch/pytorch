// cuda-shader-object-layout.cpp
#include "cuda-shader-object-layout.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

ShaderObjectLayoutImpl::ShaderObjectLayoutImpl(
    RendererBase* renderer,
    slang::ISession* session,
    slang::TypeLayoutReflection* layout)
{
    m_elementTypeLayout = _unwrapParameterGroups(layout, m_containerType);

    initBase(renderer, session, m_elementTypeLayout);

    // Compute the binding ranges that are used to store
    // the logical contents of the object in memory. These will relate
    // to the descriptor ranges in the various sets, but not always
    // in a one-to-one fashion.

    SlangInt bindingRangeCount = m_elementTypeLayout->getBindingRangeCount();
    for (SlangInt r = 0; r < bindingRangeCount; ++r)
    {
        slang::BindingType slangBindingType = m_elementTypeLayout->getBindingRangeType(r);
        SlangInt count = m_elementTypeLayout->getBindingRangeBindingCount(r);
        slang::TypeLayoutReflection* slangLeafTypeLayout =
            m_elementTypeLayout->getBindingRangeLeafTypeLayout(r);

        SlangInt descriptorSetIndex = m_elementTypeLayout->getBindingRangeDescriptorSetIndex(r);
        SlangInt rangeIndexInDescriptorSet =
            m_elementTypeLayout->getBindingRangeFirstDescriptorRangeIndex(r);

        // TODO: This logic assumes that for any binding range that might consume
        // multiple kinds of resources, the descriptor range for its uniform
        // usage will be the first one in the range.
        //
        // We need to decide whether that assumption is one we intend to support
        // applications making, or whether they should be forced to perform a
        // linear search over the descriptor ranges for a specific binding range.
        //
        auto uniformOffset = m_elementTypeLayout->getDescriptorSetDescriptorRangeIndexOffset(
            descriptorSetIndex,
            rangeIndexInDescriptorSet);

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
            baseIndex = m_resourceCount;
            m_resourceCount += count;
            break;
        default:
            baseIndex = m_resourceCount;
            m_resourceCount += count;
            break;
        }

        BindingRangeInfo bindingRangeInfo;
        bindingRangeInfo.bindingType = slangBindingType;
        bindingRangeInfo.count = count;
        bindingRangeInfo.baseIndex = baseIndex;
        bindingRangeInfo.uniformOffset = uniformOffset;
        bindingRangeInfo.subObjectIndex = subObjectIndex;
        bindingRangeInfo.isSpecializable = m_elementTypeLayout->isBindingRangeSpecializable(r);
        m_bindingRanges.add(bindingRangeInfo);
    }

    SlangInt subObjectRangeCount = m_elementTypeLayout->getSubObjectRangeCount();
    for (SlangInt r = 0; r < subObjectRangeCount; ++r)
    {
        SlangInt bindingRangeIndex = m_elementTypeLayout->getSubObjectRangeBindingRangeIndex(r);
        auto slangBindingType = m_elementTypeLayout->getBindingRangeType(bindingRangeIndex);
        slang::TypeLayoutReflection* slangLeafTypeLayout =
            m_elementTypeLayout->getBindingRangeLeafTypeLayout(bindingRangeIndex);

        // A sub-object range can either represent a sub-object of a known
        // type, like a `ConstantBuffer<Foo>` or `ParameterBlock<Foo>`
        // (in which case we can pre-compute a layout to use, based on
        // the type `Foo`) *or* it can represent a sub-object of some
        // existential type (e.g., `IBar`) in which case we cannot
        // know the appropriate type/layout of sub-object to allocate.
        //
        RefPtr<ShaderObjectLayoutImpl> subObjectLayout;
        if (slangBindingType != slang::BindingType::ExistentialValue)
        {
            subObjectLayout = new ShaderObjectLayoutImpl(
                renderer,
                session,
                slangLeafTypeLayout->getElementTypeLayout());
        }

        SubObjectRangeInfo subObjectRange;
        subObjectRange.bindingRangeIndex = bindingRangeIndex;
        subObjectRange.layout = subObjectLayout;
        subObjectRanges.add(subObjectRange);
    }
}

Index ShaderObjectLayoutImpl::getResourceCount() const
{
    return m_resourceCount;
}
Index ShaderObjectLayoutImpl::getSubObjectCount() const
{
    return m_subObjectCount;
}
List<SubObjectRangeInfo>& ShaderObjectLayoutImpl::getSubObjectRanges()
{
    return subObjectRanges;
}
BindingRangeInfo ShaderObjectLayoutImpl::getBindingRange(Index index)
{
    return m_bindingRanges[index];
}
Index ShaderObjectLayoutImpl::getBindingRangeCount() const
{
    return m_bindingRanges.getCount();
}

RootShaderObjectLayoutImpl::RootShaderObjectLayoutImpl(
    RendererBase* renderer,
    slang::ProgramLayout* inProgramLayout)
    : ShaderObjectLayoutImpl(
          renderer,
          inProgramLayout->getSession(),
          inProgramLayout->getGlobalParamsTypeLayout())
    , programLayout(inProgramLayout)
{
    for (UInt i = 0; i < programLayout->getEntryPointCount(); i++)
    {
        entryPointLayouts.add(new ShaderObjectLayoutImpl(
            renderer,
            programLayout->getSession(),
            programLayout->getEntryPointByIndex(i)->getTypeLayout()));
    }
}

int RootShaderObjectLayoutImpl::getKernelIndex(UnownedStringSlice kernelName)
{
    for (int i = 0; i < (int)programLayout->getEntryPointCount(); i++)
    {
        auto entryPoint = programLayout->getEntryPointByIndex(i);
        if (kernelName == entryPoint->getName())
        {
            return i;
        }
    }
    return -1;
}

void RootShaderObjectLayoutImpl::getKernelThreadGroupSize(int kernelIndex, UInt* threadGroupSizes)
{
    auto entryPoint = programLayout->getEntryPointByIndex(kernelIndex);
    entryPoint->getComputeThreadGroupSize(3, threadGroupSizes);
}

} // namespace cuda
#endif
} // namespace gfx
