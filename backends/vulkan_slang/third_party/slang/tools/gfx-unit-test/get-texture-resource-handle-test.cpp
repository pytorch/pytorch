#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

#if SLANG_WINDOWS_FAMILY
#include <d3d12.h>
#endif

using namespace gfx;

namespace gfx_test
{
void getTextureResourceHandleTestImpl(IDevice* device, UnitTestContext* context)
{
    ITextureResource::Desc desc = {};
    desc.type = IResource::Type::Texture2D;
    desc.numMipLevels = 1;
    desc.size.width = 1;
    desc.size.height = 1;
    desc.size.depth = 1;
    desc.defaultState = ResourceState::UnorderedAccess;
    desc.format = Format::R16G16B16A16_FLOAT;

    Slang::ComPtr<ITextureResource> buffer;
    buffer = device->createTextureResource(desc);

    InteropHandle handle;
    GFX_CHECK_CALL_ABORT(buffer->getNativeResourceHandle(&handle));
    if (device->getDeviceInfo().deviceType == gfx::DeviceType::Vulkan)
    {
        SLANG_CHECK(handle.handleValue != 0);
        SLANG_CHECK(handle.api == InteropHandleAPI::Vulkan);
    }

#if SLANG_WINDOWS_FAMILY
    else
    {
        SLANG_CHECK(handle.api == InteropHandleAPI::D3D12);
        auto d3d12Handle = (ID3D12Resource*)handle.handleValue;
        Slang::ComPtr<IUnknown> testHandle1;
        GFX_CHECK_CALL_ABORT(d3d12Handle->QueryInterface<IUnknown>(testHandle1.writeRef()));
        Slang::ComPtr<ID3D12Resource> testHandle2;
        GFX_CHECK_CALL_ABORT(testHandle1->QueryInterface<ID3D12Resource>(testHandle2.writeRef()));
        SLANG_CHECK(d3d12Handle == testHandle2.get());
    }
#endif
}

void getTextureResourceHandleTestAPI(UnitTestContext* context, Slang::RenderApiFlag::Enum api)
{
    if ((api & context->enabledApis) == 0)
    {
        SLANG_IGNORE_TEST;
    }
    Slang::ComPtr<IDevice> device;
    IDevice::Desc deviceDesc = {};
    switch (api)
    {
    case Slang::RenderApiFlag::D3D11:
        deviceDesc.deviceType = gfx::DeviceType::DirectX11;
        break;
    case Slang::RenderApiFlag::D3D12:
        deviceDesc.deviceType = gfx::DeviceType::DirectX12;
        break;
    case Slang::RenderApiFlag::Vulkan:
        deviceDesc.deviceType = gfx::DeviceType::Vulkan;
        break;
    default:
        SLANG_IGNORE_TEST;
    }
    deviceDesc.slang.slangGlobalSession = context->slangGlobalSession;
    const char* searchPaths[] = {"", "../../tools/gfx-unit-test", "tools/gfx-unit-test"};
    deviceDesc.slang.searchPathCount = (SlangInt)SLANG_COUNT_OF(searchPaths);
    deviceDesc.slang.searchPaths = searchPaths;
    auto createDeviceResult = gfxCreateDevice(&deviceDesc, device.writeRef());
    if (SLANG_FAILED(createDeviceResult))
    {
        SLANG_IGNORE_TEST;
    }
    // Ignore this test on swiftshader. Swiftshader seems to have a bug that causes the test
    // to crash.
    if (Slang::String(device->getDeviceInfo().adapterName).toLower().contains("swiftshader"))
    {
        SLANG_IGNORE_TEST;
    }
    getTextureResourceHandleTestImpl(device, context);
}

SLANG_UNIT_TEST(getTextureResourceHandleD3D12)
{
    return getTextureResourceHandleTestAPI(unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(getTextureResourceHandleVulkan)
{
    return getTextureResourceHandleTestAPI(unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
