// metal-helper-functions.h
#pragma once
#include "metal-base.h"

namespace gfx
{
using namespace Slang;
namespace metal
{

/// A "simple" binding offset that records an offset in buffer/texture/sampler slots
struct BindingOffset
{
    uint32_t buffer = 0;
    uint32_t texture = 0;
    uint32_t sampler = 0;

    /// Create a default (zero) offset
    BindingOffset() = default;

    /// Create an offset based on offset information in the given Slang `varLayout`
    BindingOffset(slang::VariableLayoutReflection* varLayout)
    {
        if (varLayout)
        {
            buffer = (uint32_t)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_METAL_BUFFER);
            texture = (uint32_t)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_METAL_TEXTURE);
            sampler = (uint32_t)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_METAL_SAMPLER);
        }
    }

    /// Create an offset based on size/stride information in the given Slang `typeLayout`
    BindingOffset(slang::TypeLayoutReflection* typeLayout)
    {
        if (typeLayout)
        {
            buffer = (uint32_t)typeLayout->getSize(SLANG_PARAMETER_CATEGORY_METAL_BUFFER);
            texture = (uint32_t)typeLayout->getSize(SLANG_PARAMETER_CATEGORY_METAL_TEXTURE);
            sampler = (uint32_t)typeLayout->getSize(SLANG_PARAMETER_CATEGORY_METAL_SAMPLER);
        }
    }

    /// Add any values in the given `offset`
    void operator+=(BindingOffset const& offset)
    {
        buffer += offset.buffer;
        texture += offset.texture;
        sampler += offset.sampler;
    }
};

/// Contextual data and operations required when binding shader objects to the pipeline state
struct BindingContext
{
    DeviceImpl* device = nullptr;
    virtual void setBuffer(MTL::Buffer* buffer, NS::UInteger index) = 0;
    virtual void setTexture(MTL::Texture* texture, NS::UInteger index) = 0;
    virtual void setSampler(MTL::SamplerState* sampler, NS::UInteger index) = 0;
    virtual void useResources(
        MTL::Resource const** resources,
        NS::UInteger count,
        MTL::ResourceUsage usage) = 0;
};

struct ComputeBindingContext : public BindingContext
{
    MTL::ComputeCommandEncoder* encoder;

    Result init(DeviceImpl* device, MTL::ComputeCommandEncoder* encoder)
    {
        this->device = device;
        this->encoder = encoder;
        return SLANG_OK;
    }

    void setBuffer(MTL::Buffer* buffer, NS::UInteger index) override
    {
        encoder->setBuffer(buffer, 0, index);
    }

    void setTexture(MTL::Texture* texture, NS::UInteger index) override
    {
        encoder->setTexture(texture, index);
    }

    void setSampler(MTL::SamplerState* sampler, NS::UInteger index) override
    {
        encoder->setSamplerState(sampler, index);
    }

    void useResources(MTL::Resource const** resources, NS::UInteger count, MTL::ResourceUsage usage)
        override
    {
        encoder->useResources(resources, count, usage);
    }
};

struct RenderBindingContext : public BindingContext
{
    MTL::RenderCommandEncoder* encoder;

    Result init(DeviceImpl* device, MTL::RenderCommandEncoder* encoder)
    {
        this->device = device;
        this->encoder = encoder;
        return SLANG_OK;
    }

    void setBuffer(MTL::Buffer* buffer, NS::UInteger index) override
    {
        encoder->setVertexBuffer(buffer, 0, index);
        encoder->setFragmentBuffer(buffer, 0, index);
    }

    void setTexture(MTL::Texture* texture, NS::UInteger index) override
    {
        encoder->setVertexTexture(texture, index);
        encoder->setFragmentTexture(texture, index);
    }

    void setSampler(MTL::SamplerState* sampler, NS::UInteger index) override
    {
        encoder->setVertexSamplerState(sampler, index);
        encoder->setFragmentSamplerState(sampler, index);
    }

    void useResources(MTL::Resource const** resources, NS::UInteger count, MTL::ResourceUsage usage)
        override
    {
        encoder->useResources(resources, count, usage);
    }
};

} // namespace metal

Result SLANG_MCALL getMetalAdapters(List<AdapterInfo>& outAdapters);
Result SLANG_MCALL createMetalDevice(const IDevice::Desc* desc, IDevice** outRenderer);

} // namespace gfx
