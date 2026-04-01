// cuda-helper-functions.cpp
#include "cuda-helper-functions.h"

#include "cuda-device.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{
SlangResult CUDAErrorInfo::handle() const
{
    StringBuilder builder;
    builder << "Error: " << m_filePath << " (" << m_lineNo << ") :";

    if (m_errorName)
    {
        builder << m_errorName << " : ";
    }
    if (m_errorString)
    {
        builder << m_errorString;
    }

    getDebugCallback()->handleMessage(
        DebugMessageType::Error,
        DebugMessageSource::Driver,
        builder.getUnownedSlice().begin());

    // Slang::signalUnexpectedError(builder.getBuffer());
    return SLANG_FAIL;
}

SlangResult _handleCUDAError(CUresult cuResult, const char* file, int line)
{
    CUDAErrorInfo info(file, line);
    cuGetErrorString(cuResult, &info.m_errorString);
    cuGetErrorName(cuResult, &info.m_errorName);
    return info.handle();
}

#ifdef RENDER_TEST_OPTIX

static bool _isError(OptixResult result)
{
    return result != OPTIX_SUCCESS;
}

#if 1
static SlangResult _handleOptixError(OptixResult result, char const* file, int line)
{
    fprintf(
        stderr,
        "%s(%d): optix: %s (%s)\n",
        file,
        line,
        optixGetErrorString(result),
        optixGetErrorName(result));
    return SLANG_FAIL;
}

void _optixLogCallback(unsigned int level, const char* tag, const char* message, void* userData)
{
    fprintf(stderr, "optix: %s (%s)\n", message, tag);
}
#endif
#endif

AdapterLUID getAdapterLUID(int deviceIndex)
{
    CUdevice device;
    cuDeviceGet(&device, deviceIndex);
    AdapterLUID luid = {};
    unsigned int deviceNodeMask;
    cuDeviceGetLuid((char*)&luid, &deviceNodeMask, device);
    return luid;
}

Result SLANG_MCALL getAdapters(List<AdapterInfo>& outAdapters)
{
    int deviceCount;
    SLANG_CUDA_RETURN_ON_FAIL(cuDeviceGetCount(&deviceCount));
    for (int deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++)
    {
        CUdevice device;
        SLANG_CUDA_RETURN_ON_FAIL(cuDeviceGet(&device, deviceIndex));

        AdapterInfo info = {};
        SLANG_CUDA_RETURN_ON_FAIL(cuDeviceGetName(info.name, sizeof(info.name), device));
        info.luid = getAdapterLUID(deviceIndex);
        outAdapters.add(info);
    }

    return SLANG_OK;
}

} // namespace cuda

Result SLANG_MCALL getCUDAAdapters(List<AdapterInfo>& outAdapters)
{
    return cuda::getAdapters(outAdapters);
}

Result SLANG_MCALL createCUDADevice(const IDevice::Desc* desc, IDevice** outDevice)
{
    RefPtr<cuda::DeviceImpl> result = new cuda::DeviceImpl();
    SLANG_RETURN_ON_FAIL(result->initialize(*desc));
    returnComPtr(outDevice, result);
    return SLANG_OK;
}
#else

Result SLANG_MCALL getCUDAAdapters(List<AdapterInfo>& outAdapters)
{
    SLANG_UNUSED(outAdapters);
    return SLANG_FAIL;
}

Result SLANG_MCALL createCUDADevice(const IDevice::Desc* desc, IDevice** outDevice)
{
    SLANG_UNUSED(desc);
    *outDevice = nullptr;
    return SLANG_FAIL;
}
#endif

} // namespace gfx
