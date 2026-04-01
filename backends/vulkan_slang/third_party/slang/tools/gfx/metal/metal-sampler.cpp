// metal-sampler.cpp
#include "metal-sampler.h"

#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

SamplerStateImpl::~SamplerStateImpl() {}

Result SamplerStateImpl::init(DeviceImpl* device, const ISamplerState::Desc& desc)
{
    m_device = device;

    NS::SharedPtr<MTL::SamplerDescriptor> samplerDesc =
        NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());

    samplerDesc->setMinFilter(MetalUtil::translateSamplerMinMagFilter(desc.minFilter));
    samplerDesc->setMagFilter(MetalUtil::translateSamplerMinMagFilter(desc.magFilter));
    samplerDesc->setMipFilter(MetalUtil::translateSamplerMipFilter(desc.mipFilter));

    samplerDesc->setSAddressMode(MetalUtil::translateSamplerAddressMode(desc.addressU));
    samplerDesc->setTAddressMode(MetalUtil::translateSamplerAddressMode(desc.addressV));
    samplerDesc->setRAddressMode(MetalUtil::translateSamplerAddressMode(desc.addressW));

    samplerDesc->setMaxAnisotropy(Math::Clamp(desc.maxAnisotropy, 1u, 16u));

    // TODO: support translation of border color...
    MTL::SamplerBorderColor borderColor = MTL::SamplerBorderColorOpaqueBlack;
    samplerDesc->setBorderColor(borderColor);

    samplerDesc->setNormalizedCoordinates(true);

    samplerDesc->setCompareFunction(MetalUtil::translateCompareFunction(desc.comparisonFunc));
    samplerDesc->setLodMinClamp(Math::Clamp(desc.minLOD, 0.f, 1000.f));
    samplerDesc->setLodMaxClamp(Math::Clamp(desc.maxLOD, samplerDesc->lodMinClamp(), 1000.f));

    samplerDesc->setSupportArgumentBuffers(true);

    // TODO: no support for reduction op

    m_samplerState = NS::TransferPtr(m_device->m_device->newSamplerState(samplerDesc.get()));

    return m_samplerState ? SLANG_OK : SLANG_FAIL;
}

Result SamplerStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Metal;
    outHandle->handleValue = reinterpret_cast<intptr_t>(m_samplerState.get());
    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
