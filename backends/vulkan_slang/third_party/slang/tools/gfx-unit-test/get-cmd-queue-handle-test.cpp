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
void getQueueHandleTestImpl(IDevice* device, UnitTestContext* context)
{
    ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
    auto queue = device->createCommandQueue(queueDesc);
    InteropHandle handle;
    GFX_CHECK_CALL_ABORT(queue->getNativeHandle(&handle));
    if (device->getDeviceInfo().deviceType == gfx::DeviceType::Vulkan)
    {
        SLANG_CHECK(handle.handleValue != 0);
    }
#if SLANG_WINDOWS_FAMILY
    else
    {
        auto d3d12Queue = (ID3D12CommandQueue*)handle.handleValue;
        Slang::ComPtr<IUnknown> testHandle1;
        GFX_CHECK_CALL_ABORT(d3d12Queue->QueryInterface<IUnknown>(testHandle1.writeRef()));
        Slang::ComPtr<ID3D12CommandQueue> testHandle2;
        GFX_CHECK_CALL_ABORT(
            testHandle1->QueryInterface<ID3D12CommandQueue>(testHandle2.writeRef()));
        SLANG_CHECK(d3d12Queue == testHandle2.get());
    }
#endif
}

void getQueueHandleTestAPI(UnitTestContext* context, Slang::RenderApiFlag::Enum api)
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
    getQueueHandleTestImpl(device, context);
}

SLANG_UNIT_TEST(getCmdQueueHandleD3D12)
{
    return getQueueHandleTestAPI(unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(getCmdQueueHandleVulkan)
{
    return getQueueHandleTestAPI(unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
