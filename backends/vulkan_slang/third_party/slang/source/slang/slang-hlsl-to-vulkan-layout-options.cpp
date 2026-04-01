// slang-hlsl-to-vulkan-layout-options.cpp

#include "slang-hlsl-to-vulkan-layout-options.h"

#include "slang-compiler-options.h"

namespace Slang
{

namespace
{ // anonymous

typedef HLSLToVulkanLayoutOptions::Kind ShiftKind;

/* {b|s|t|u}

https://github.com/KhronosGroup/glslang/wiki/HLSL-FAQ
*/
static NamesDescriptionValue s_vulkanShiftKinds[] = {
    {ValueInt(ShiftKind::ConstantBuffer), "b", "Constant buffer view"},
    {ValueInt(ShiftKind::Sampler), "s", "Sampler"},
    {ValueInt(ShiftKind::ShaderResource), "t", "Shader resource view"},
    {ValueInt(ShiftKind::UnorderedAccess), "u", "Unorderd access view"},
};

} // namespace

/* static */ ConstArrayView<NamesDescriptionValue> HLSLToVulkanLayoutOptions::getKindInfos()
{
    return makeConstArrayView(s_vulkanShiftKinds);
}

void HLSLToVulkanLayoutOptions::loadFromOptionSet(CompilerOptionSet& optionSet)
{
    auto allShift = optionSet.getArray(CompilerOptionName::VulkanBindShiftAll);
    for (auto v : allShift)
    {
        setAllShift((Kind)v.intValue, v.intValue2);
    }
    auto shifts = optionSet.getArray(CompilerOptionName::VulkanBindShift);
    for (auto v : shifts)
    {
        uint8_t kind;
        int set;
        int shift;
        v.unpackInt3(kind, set, shift);
        setShift((Kind)kind, set, shift);
    }
    m_emitSPIRVReflectionInfo = optionSet.getBoolOption(CompilerOptionName::VulkanEmitReflection);
    if (auto bindGlobals = optionSet.options.tryGetValue(CompilerOptionName::VulkanBindGlobals))
    {
        if (bindGlobals->getCount())
        {
            m_globalsBinding.index = (*bindGlobals)[0].intValue;
            m_globalsBinding.set = (*bindGlobals)[0].intValue2;
        }
    }
    m_useGLLayout = optionSet.getBoolOption(CompilerOptionName::VulkanUseGLLayout);
    m_useOriginalEntryPointName =
        optionSet.getBoolOption(CompilerOptionName::VulkanUseEntryPointName);
}

HLSLToVulkanLayoutOptions::HLSLToVulkanLayoutOptions()
{
    // Clear the all shifts
    for (auto& shift : m_allShifts)
    {
        shift = kInvalidShift;
    }

    SLANG_ASSERT(isReset());
}

void HLSLToVulkanLayoutOptions::setGlobalsBinding(const Binding& binding)
{
    m_globalsBinding = binding;
}

void HLSLToVulkanLayoutOptions::reset()
{
    m_kindShiftEnabledFlags = 0;

    for (auto& shift : m_allShifts)
    {
        shift = kInvalidShift;
    }

    m_shifts.clear();
}

void HLSLToVulkanLayoutOptions::setAllShift(Kind kind, Index shift)
{
    SLANG_ASSERT(shift != kInvalidShift);

    m_allShifts[Index(kind)] = shift;
    _enableShiftForKind(kind);
}

void HLSLToVulkanLayoutOptions::setShift(Kind kind, Index set, Index shift)
{
    SLANG_ASSERT(shift != kInvalidShift);

    Key key{kind, set};
    m_shifts.set(key, shift);
    _enableShiftForKind(kind);
}

Index HLSLToVulkanLayoutOptions::getShift(Kind kind, Index set) const
{
    if (canInferBindingForKind(kind))
    {
        // We lookup a shift for a set first as this shift is "more specific" and
        // is seen as taken precedent over the "all" scenario
        if (auto ptr = m_shifts.tryGetValue(Key{kind, set}))
        {
            return *ptr;
        }

        // Must have an `all` shift
        return m_allShifts[Index(kind)];
    }
    return kInvalidShift;
}

bool HLSLToVulkanLayoutOptions::hasState() const
{
    return canInferBindings() || hasGlobalsBinding() || getUseOriginalEntryPointName() ||
           shouldUseGLLayout() || shouldEmitSPIRVReflectionInfo();
}

HLSLToVulkanLayoutOptions::Binding HLSLToVulkanLayoutOptions::inferBinding(
    Kind kind,
    const Binding& inBinding) const
{
    auto shift = getShift(kind, inBinding.set);

    if (shift != kInvalidShift)
    {
        Binding binding(inBinding);
        binding.index += shift;
        return binding;
    }

    // Else return an invalid binding
    return Binding();
}

/* static */ HLSLToVulkanLayoutOptions::Kind HLSLToVulkanLayoutOptions::getKind(
    slang::ParameterCategory param)
{
    typedef slang::ParameterCategory ParameterCategory;

    switch (param)
    {
    case ParameterCategory::Mixed:
        {
            // TODO(JS):
            // Hmm, is this TextureSampler?
            return Kind::Invalid;
        }
    case ParameterCategory::Uniform:
    case ParameterCategory::ConstantBuffer:
        {
            return Kind::ConstantBuffer;
        }
    case ParameterCategory::ShaderResource:
        return Kind::ShaderResource;
    case ParameterCategory::UnorderedAccess:
        return Kind::UnorderedAccess;
    case ParameterCategory::SamplerState:
        return Kind::Sampler;

    default:
        {
            return Kind::Invalid;
        }
    }
}

} // namespace Slang
