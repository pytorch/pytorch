// metal-helper-functions.cpp
#include "metal-helper-functions.h"

#include "metal-device.h"
#include "metal-util.h"

namespace gfx
{

using namespace Slang;

Result SLANG_MCALL getMetalAdapters(List<AdapterInfo>& outAdapters)
{
    AUTORELEASEPOOL

    auto addAdapter = [&](MTL::Device* device)
    {
        AdapterInfo info = {};
        const char* name = device->name()->cString(NS::ASCIIStringEncoding);
        memcpy(info.name, name, Math::Min(strlen(name), sizeof(AdapterInfo::name) - 1));
        uint64_t registryID = device->registryID();
        memcpy(&info.luid.luid[0], &registryID, sizeof(registryID));
        outAdapters.add(info);
    };

    NS::Array* devices = MTL::CopyAllDevices();
    if (devices->count() > 0)
    {
        for (int i = 0; i < devices->count(); ++i)
        {
            MTL::Device* device = static_cast<MTL::Device*>(devices->object(i));
            addAdapter(device);
        }
    }
    else
    {
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        addAdapter(device);
        device->release();
    }
    return SLANG_OK;
}

Result SLANG_MCALL createMetalDevice(const IDevice::Desc* desc, IDevice** outRenderer)
{
    RefPtr<metal::DeviceImpl> result = new metal::DeviceImpl();
    SLANG_RETURN_ON_FAIL(result->initialize(*desc));
    returnComPtr(outRenderer, result);
    return SLANG_OK;
}

} // namespace gfx
