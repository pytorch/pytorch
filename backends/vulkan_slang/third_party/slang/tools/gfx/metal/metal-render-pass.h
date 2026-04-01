// metal-render-pass.h
#pragma once

#include "metal-base.h"
#include "metal-device.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class RenderPassLayoutImpl : public IRenderPassLayout, public ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    IRenderPassLayout* getInterface(const Guid& guid);

public:
    RefPtr<DeviceImpl> m_device;
    NS::SharedPtr<MTL::RenderPassDescriptor> m_renderPassDesc;

    Result init(DeviceImpl* device, const IRenderPassLayout::Desc& desc);
};

} // namespace metal
} // namespace gfx
