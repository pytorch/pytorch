// render.cpp
#include "../../source/core/slang-blob.h"
#include "../../source/core/slang-math.h"
#include "debug-layer/debug-device.h"
#include "open-gl/render-gl.h"
#include "renderer-shared.h"

#include <cstring>

namespace gfx
{
using namespace Slang;

Result SLANG_MCALL createD3D11Device(const IDevice::Desc* desc, IDevice** outDevice);
Result SLANG_MCALL createD3D12Device(const IDevice::Desc* desc, IDevice** outDevice);
Result SLANG_MCALL createVKDevice(const IDevice::Desc* desc, IDevice** outDevice);
Result SLANG_MCALL createMetalDevice(const IDevice::Desc* desc, IDevice** outDevice);
Result SLANG_MCALL createCUDADevice(const IDevice::Desc* desc, IDevice** outDevice);
Result SLANG_MCALL createCPUDevice(const IDevice::Desc* desc, IDevice** outDevice);

Result SLANG_MCALL getD3D11Adapters(List<AdapterInfo>& outAdapters);
Result SLANG_MCALL getD3D12Adapters(List<AdapterInfo>& outAdapters);
Result SLANG_MCALL getVKAdapters(List<AdapterInfo>& outAdapters);
Result SLANG_MCALL getMetalAdapters(List<AdapterInfo>& outAdapters);
Result SLANG_MCALL getCUDAAdapters(List<AdapterInfo>& outAdapters);

Result SLANG_MCALL reportD3DLiveObjects();

static bool debugLayerEnabled = false;
bool isGfxDebugLayerEnabled()
{
    return debugLayerEnabled;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global Renderer Functions !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#define GFX_FORMAT_SIZE(name, blockSizeInBytes, pixelsPerBlock) {blockSizeInBytes, pixelsPerBlock},

static const uint32_t s_formatSizeInfo[][2] = {GFX_FORMAT(GFX_FORMAT_SIZE)};

static bool _checkFormat()
{
    Index value = 0;
    Index count = 0;

    // Check the values are in the same order
#define GFX_FORMAT_CHECK(name, blockSizeInBytes, pixelsPerblock) \
    count += Index(Index(Format::name) == value++);
    GFX_FORMAT(GFX_FORMAT_CHECK)

    const bool r = (count == Index(Format::_Count));
    SLANG_ASSERT(r);
    return r;
}

// We don't make static because we will get a warning that it's unused
static const bool _checkFormatResult = _checkFormat();

struct FormatInfoMap
{
    FormatInfoMap()
    {
        // Set all to nothing initially
        for (auto& info : m_infos)
        {
            info.channelCount = 0;
            info.channelType = SLANG_SCALAR_TYPE_NONE;
        }

        set(Format::R32G32B32A32_TYPELESS, SLANG_SCALAR_TYPE_UINT32, 4);
        set(Format::R32G32B32_TYPELESS, SLANG_SCALAR_TYPE_UINT32, 3);
        set(Format::R32G32_TYPELESS, SLANG_SCALAR_TYPE_UINT32, 2);
        set(Format::R32_TYPELESS, SLANG_SCALAR_TYPE_UINT32, 1);

        set(Format::R16G16B16A16_TYPELESS, SLANG_SCALAR_TYPE_UINT16, 4);
        set(Format::R16G16_TYPELESS, SLANG_SCALAR_TYPE_UINT16, 2);
        set(Format::R16_TYPELESS, SLANG_SCALAR_TYPE_UINT16, 1);

        set(Format::R8G8B8A8_TYPELESS, SLANG_SCALAR_TYPE_UINT8, 4);
        set(Format::R8G8_TYPELESS, SLANG_SCALAR_TYPE_UINT8, 2);
        set(Format::R8_TYPELESS, SLANG_SCALAR_TYPE_UINT8, 1);
        set(Format::B8G8R8A8_TYPELESS, SLANG_SCALAR_TYPE_UINT8, 4);

        set(Format::R32G32B32A32_FLOAT, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R32G32B32_FLOAT, SLANG_SCALAR_TYPE_FLOAT32, 3);
        set(Format::R32G32_FLOAT, SLANG_SCALAR_TYPE_FLOAT32, 2);
        set(Format::R32_FLOAT, SLANG_SCALAR_TYPE_FLOAT32, 1);

        set(Format::R16G16B16A16_FLOAT, SLANG_SCALAR_TYPE_FLOAT16, 4);
        set(Format::R16G16_FLOAT, SLANG_SCALAR_TYPE_FLOAT16, 2);
        set(Format::R16_FLOAT, SLANG_SCALAR_TYPE_FLOAT16, 1);

        set(Format::R64_UINT, SLANG_SCALAR_TYPE_UINT64, 1);

        set(Format::R32G32B32A32_UINT, SLANG_SCALAR_TYPE_UINT32, 4);
        set(Format::R32G32B32_UINT, SLANG_SCALAR_TYPE_UINT32, 3);
        set(Format::R32G32_UINT, SLANG_SCALAR_TYPE_UINT32, 2);
        set(Format::R32_UINT, SLANG_SCALAR_TYPE_UINT32, 1);

        set(Format::R16G16B16A16_UINT, SLANG_SCALAR_TYPE_UINT16, 4);
        set(Format::R16G16_UINT, SLANG_SCALAR_TYPE_UINT16, 2);
        set(Format::R16_UINT, SLANG_SCALAR_TYPE_UINT16, 1);

        set(Format::R8G8B8A8_UINT, SLANG_SCALAR_TYPE_UINT8, 4);
        set(Format::R8G8_UINT, SLANG_SCALAR_TYPE_UINT8, 2);
        set(Format::R8_UINT, SLANG_SCALAR_TYPE_UINT8, 1);

        set(Format::R64_SINT, SLANG_SCALAR_TYPE_INT64, 1);

        set(Format::R32G32B32A32_SINT, SLANG_SCALAR_TYPE_INT32, 4);
        set(Format::R32G32B32_SINT, SLANG_SCALAR_TYPE_INT32, 3);
        set(Format::R32G32_SINT, SLANG_SCALAR_TYPE_INT32, 2);
        set(Format::R32_SINT, SLANG_SCALAR_TYPE_INT32, 1);

        set(Format::R16G16B16A16_SINT, SLANG_SCALAR_TYPE_INT16, 4);
        set(Format::R16G16_SINT, SLANG_SCALAR_TYPE_INT16, 2);
        set(Format::R16_SINT, SLANG_SCALAR_TYPE_INT16, 1);

        set(Format::R8G8B8A8_SINT, SLANG_SCALAR_TYPE_INT8, 4);
        set(Format::R8G8_SINT, SLANG_SCALAR_TYPE_INT8, 2);
        set(Format::R8_SINT, SLANG_SCALAR_TYPE_INT8, 1);

        set(Format::R16G16B16A16_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R16G16_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 2);
        set(Format::R16_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 1);

        set(Format::R8G8B8A8_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R8G8B8A8_UNORM_SRGB, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R8G8_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 2);
        set(Format::R8_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 1);
        set(Format::B8G8R8A8_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::B8G8R8A8_UNORM_SRGB, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::B8G8R8X8_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::B8G8R8X8_UNORM_SRGB, SLANG_SCALAR_TYPE_FLOAT32, 4);

        set(Format::R16G16B16A16_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R16G16_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 2);
        set(Format::R16_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 1);

        set(Format::R8G8B8A8_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R8G8_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 2);
        set(Format::R8_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 1);

        set(Format::D32_FLOAT, SLANG_SCALAR_TYPE_FLOAT32, 1);
        set(Format::D16_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 1);
        set(Format::D32_FLOAT_S8_UINT, SLANG_SCALAR_TYPE_FLOAT32, 2);
        set(Format::R32_FLOAT_X32_TYPELESS, SLANG_SCALAR_TYPE_FLOAT32, 2);

        set(Format::B4G4R4A4_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::B5G6R5_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 3);
        set(Format::B5G5R5A1_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);

        set(Format::R9G9B9E5_SHAREDEXP, SLANG_SCALAR_TYPE_FLOAT32, 3);
        set(Format::R10G10B10A2_TYPELESS, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R10G10B10A2_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4);
        set(Format::R10G10B10A2_UINT, SLANG_SCALAR_TYPE_UINT32, 4);
        set(Format::R11G11B10_FLOAT, SLANG_SCALAR_TYPE_FLOAT32, 3);

        set(Format::BC1_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
        set(Format::BC1_UNORM_SRGB, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
        set(Format::BC2_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
        set(Format::BC2_UNORM_SRGB, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
        set(Format::BC3_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
        set(Format::BC3_UNORM_SRGB, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
        set(Format::BC4_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 1, 4, 4);
        set(Format::BC4_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 1, 4, 4);
        set(Format::BC5_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 2, 4, 4);
        set(Format::BC5_SNORM, SLANG_SCALAR_TYPE_FLOAT32, 2, 4, 4);
        set(Format::BC6H_UF16, SLANG_SCALAR_TYPE_FLOAT32, 3, 4, 4);
        set(Format::BC6H_SF16, SLANG_SCALAR_TYPE_FLOAT32, 3, 4, 4);
        set(Format::BC7_UNORM, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
        set(Format::BC7_UNORM_SRGB, SLANG_SCALAR_TYPE_FLOAT32, 4, 4, 4);
    }

    void set(
        Format format,
        SlangScalarType type,
        Index channelCount,
        uint32_t blockWidth = 1,
        uint32_t blockHeight = 1)
    {
        FormatInfo& info = m_infos[Index(format)];
        info.channelCount = uint8_t(channelCount);
        info.channelType = uint8_t(type);

        auto sizeInfo = s_formatSizeInfo[Index(format)];
        info.blockSizeInBytes = sizeInfo[0];
        info.pixelsPerBlock = sizeInfo[1];
        info.blockWidth = blockWidth;
        info.blockHeight = blockHeight;
    }

    const FormatInfo& get(Format format) const { return m_infos[Index(format)]; }

    FormatInfo m_infos[Index(Format::_Count)];
};

static const FormatInfoMap s_formatInfoMap;

static void _compileTimeAsserts()
{
    SLANG_COMPILE_TIME_ASSERT(SLANG_COUNT_OF(s_formatSizeInfo) == int(Format::_Count));
}

extern "C"
{
    SLANG_GFX_API bool SLANG_MCALL gfxIsCompressedFormat(Format format)
    {
        switch (format)
        {
        case Format::BC1_UNORM:
        case Format::BC1_UNORM_SRGB:
        case Format::BC2_UNORM:
        case Format::BC2_UNORM_SRGB:
        case Format::BC3_UNORM:
        case Format::BC3_UNORM_SRGB:
        case Format::BC4_UNORM:
        case Format::BC4_SNORM:
        case Format::BC5_UNORM:
        case Format::BC5_SNORM:
        case Format::BC6H_UF16:
        case Format::BC6H_SF16:
        case Format::BC7_UNORM:
        case Format::BC7_UNORM_SRGB:
            return true;
        default:
            return false;
        }
    }

    SLANG_GFX_API bool SLANG_MCALL gfxIsTypelessFormat(Format format)
    {
        switch (format)
        {
        case Format::R32G32B32A32_TYPELESS:
        case Format::R32G32B32_TYPELESS:
        case Format::R32G32_TYPELESS:
        case Format::R32_TYPELESS:
        case Format::R16G16B16A16_TYPELESS:
        case Format::R16G16_TYPELESS:
        case Format::R16_TYPELESS:
        case Format::R8G8B8A8_TYPELESS:
        case Format::R8G8_TYPELESS:
        case Format::R8_TYPELESS:
        case Format::B8G8R8A8_TYPELESS:
        case Format::R10G10B10A2_TYPELESS:
            return true;
        default:
            return false;
        }
    }

    SLANG_GFX_API SlangResult SLANG_MCALL gfxGetFormatInfo(Format format, FormatInfo* outInfo)
    {
        *outInfo = s_formatInfoMap.get(format);
        return SLANG_OK;
    }

    SLANG_GFX_API SlangResult SLANG_MCALL
    gfxGetAdapters(DeviceType type, ISlangBlob** outAdaptersBlob)
    {
        List<AdapterInfo> adapters;

        switch (type)
        {
#if SLANG_ENABLE_DIRECTX
        case DeviceType::DirectX11:
            SLANG_RETURN_ON_FAIL(getD3D11Adapters(adapters));
            break;
        case DeviceType::DirectX12:
            SLANG_RETURN_ON_FAIL(getD3D12Adapters(adapters));
            break;
#endif
#if SLANG_WINDOWS_FAMILY
        case DeviceType::OpenGl:
            return SLANG_E_NOT_IMPLEMENTED;
#endif
#if SLANG_WINDOWS_FAMILY || SLANG_LINUX_FAMILY
        // Assume no Vulkan or CUDA on MacOS or Cygwin
        case DeviceType::Vulkan:
            SLANG_RETURN_ON_FAIL(getVKAdapters(adapters));
            break;
        case DeviceType::CUDA:
            SLANG_RETURN_ON_FAIL(getCUDAAdapters(adapters));
            break;
#endif
#if SLANG_APPLE_FAMILY
        case DeviceType::Vulkan:
            SLANG_RETURN_ON_FAIL(getVKAdapters(adapters));
            break;
        case DeviceType::Metal:
            SLANG_RETURN_ON_FAIL(getMetalAdapters(adapters));
            break;
#endif
        case DeviceType::CPU:
            return SLANG_E_NOT_IMPLEMENTED;
        default:
            return SLANG_E_INVALID_ARG;
        }

        auto adaptersBlob =
            RawBlob::create(adapters.getBuffer(), adapters.getCount() * sizeof(AdapterInfo));
        if (outAdaptersBlob)
            returnComPtr(outAdaptersBlob, adaptersBlob);

        return SLANG_OK;
    }

    SlangResult _createDevice(const IDevice::Desc* desc, IDevice** outDevice)
    {
        switch (desc->deviceType)
        {
#if SLANG_ENABLE_DIRECTX
        case DeviceType::DirectX11:
            {
                return createD3D11Device(desc, outDevice);
            }
        case DeviceType::DirectX12:
            {
                return createD3D12Device(desc, outDevice);
            }
#endif
#if SLANG_WINDOWS_FAMILY
        case DeviceType::OpenGl:
            {
                return createGLDevice(desc, outDevice);
            }
        case DeviceType::Vulkan:
            {
                return createVKDevice(desc, outDevice);
            }
        case DeviceType::Default:
            {
                IDevice::Desc newDesc = *desc;
                newDesc.deviceType = DeviceType::DirectX12;
                if (_createDevice(&newDesc, outDevice) == SLANG_OK)
                    return SLANG_OK;
                newDesc.deviceType = DeviceType::Vulkan;
                if (_createDevice(&newDesc, outDevice) == SLANG_OK)
                    return SLANG_OK;
                newDesc.deviceType = DeviceType::DirectX11;
                if (_createDevice(&newDesc, outDevice) == SLANG_OK)
                    return SLANG_OK;
                newDesc.deviceType = DeviceType::OpenGl;
                if (_createDevice(&newDesc, outDevice) == SLANG_OK)
                    return SLANG_OK;
                return SLANG_FAIL;
            }
            break;
#elif SLANG_APPLE_FAMILY
        case DeviceType::Vulkan:
            {
                return createVKDevice(desc, outDevice);
            }
        case DeviceType::Metal:
            {
                return createMetalDevice(desc, outDevice);
            }
        case DeviceType::Default:
            {
                IDevice::Desc newDesc = *desc;
                newDesc.deviceType = DeviceType::Metal;
                if (_createDevice(&newDesc, outDevice) == SLANG_OK)
                    return SLANG_OK;
                newDesc.deviceType = DeviceType::Vulkan;
                if (_createDevice(&newDesc, outDevice) == SLANG_OK)
                    return SLANG_OK;
                return SLANG_FAIL;
            }
#elif SLANG_LINUX_FAMILY && !defined(__CYGWIN__)
        case DeviceType::Vulkan:
            {
                return createVKDevice(desc, outDevice);
            }
        case DeviceType::Default:
            {
                IDevice::Desc newDesc = *desc;
                newDesc.deviceType = DeviceType::Vulkan;
                if (_createDevice(&newDesc, outDevice) == SLANG_OK)
                    return SLANG_OK;
                return SLANG_FAIL;
            }
#endif
        case DeviceType::CUDA:
            {
                return createCUDADevice(desc, outDevice);
            }
        case DeviceType::CPU:
            {
                return createCPUDevice(desc, outDevice);
            }
            break;

        default:
            return SLANG_FAIL;
        }
    }

    SLANG_GFX_API SlangResult SLANG_MCALL
    gfxCreateDevice(const IDevice::Desc* desc, IDevice** outDevice)
    {
        ComPtr<IDevice> innerDevice;
        auto resultCode = _createDevice(desc, innerDevice.writeRef());
        if (SLANG_FAILED(resultCode))
            return resultCode;
        if (!debugLayerEnabled)
        {
            returnComPtr(outDevice, innerDevice);
            return resultCode;
        }
        RefPtr<debug::DebugDevice> debugDevice = new debug::DebugDevice();
        debugDevice->baseObject = innerDevice;
        returnComPtr(outDevice, debugDevice);
        return resultCode;
    }

    SLANG_GFX_API SlangResult SLANG_MCALL gfxReportLiveObjects()
    {
#if SLANG_ENABLE_DIRECTX
        SLANG_RETURN_ON_FAIL(reportD3DLiveObjects());
#endif
        return SLANG_OK;
    }

    SLANG_GFX_API SlangResult SLANG_MCALL gfxSetDebugCallback(IDebugCallback* callback)
    {
        _getDebugCallback() = callback;
        return SLANG_OK;
    }

    SLANG_GFX_API void SLANG_MCALL gfxEnableDebugLayer()
    {
        debugLayerEnabled = true;
    }

    const char* SLANG_MCALL gfxGetDeviceTypeName(DeviceType type)
    {
        switch (type)
        {
        case gfx::DeviceType::Unknown:
            return "Unknown";
        case gfx::DeviceType::Default:
            return "Default";
        case gfx::DeviceType::DirectX11:
            return "DirectX11";
        case gfx::DeviceType::DirectX12:
            return "DirectX12";
        case gfx::DeviceType::OpenGl:
            return "OpenGL";
        case gfx::DeviceType::Vulkan:
            return "Vulkan";
        case gfx::DeviceType::Metal:
            return "Metal";
        case gfx::DeviceType::CPU:
            return "CPU";
        case gfx::DeviceType::CUDA:
            return "CUDA";
        default:
            return "?";
        }
    }


    void SLANG_MCALL gfxGetIdentityProjection(ProjectionStyle style, float projMatrix[16])
    {
        switch (style)
        {
        case ProjectionStyle::DirectX:
        case ProjectionStyle::OpenGl:
            {
                static const float kIdentity[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
                ::memcpy(projMatrix, kIdentity, sizeof(kIdentity));
                break;
            }
        case ProjectionStyle::Vulkan:
            {
                static const float kIdentity[] = {1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
                ::memcpy(projMatrix, kIdentity, sizeof(kIdentity));
                break;
            }
        default:
            {
                assert(!"Not handled");
            }
        }
    }
}

} // namespace gfx
