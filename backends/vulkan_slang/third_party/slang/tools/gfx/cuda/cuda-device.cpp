// cuda-device.cpp
#include "cuda-device.h"

#include "cuda-buffer.h"
#include "cuda-command-queue.h"
#include "cuda-pipeline-state.h"
#include "cuda-query.h"
#include "cuda-resource-views.h"
#include "cuda-shader-object-layout.h"
#include "cuda-shader-object.h"
#include "cuda-shader-program.h"
#include "cuda-texture.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

int DeviceImpl::_calcSMCountPerMultiProcessor(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    struct SMInfo
    {
        int sm; // 0xMm (hexadecimal notation), M = SM Major version, and m = SM minor version
        int coreCount;
    };

    static const SMInfo infos[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64}};

    const int sm = ((major << 4) + minor);
    for (Index i = 0; i < SLANG_COUNT_OF(infos); ++i)
    {
        if (infos[i].sm == sm)
        {
            return infos[i].coreCount;
        }
    }

    const auto& last = infos[SLANG_COUNT_OF(infos) - 1];

    // It must be newer presumably
    SLANG_ASSERT(sm > last.sm);

    // Default to the last entry
    return last.coreCount;
}

SlangResult DeviceImpl::_findMaxFlopsDeviceIndex(int* outDeviceIndex)
{
    int smPerMultiproc = 0;
    int maxPerfDevice = -1;
    int deviceCount = 0;
    int devicesProhibited = 0;

    uint64_t maxComputePerf = 0;
    SLANG_CUDA_RETURN_ON_FAIL(cuDeviceGetCount(&deviceCount));

    // Find the best CUDA capable GPU device
    for (int currentDevice = 0; currentDevice < deviceCount; ++currentDevice)
    {
        CUdevice device;
        SLANG_CUDA_RETURN_ON_FAIL(cuDeviceGet(&device, currentDevice));
        int computeMode = -1, major = 0, minor = 0;
        SLANG_CUDA_RETURN_ON_FAIL(
            cuDeviceGetAttribute(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device));
        SLANG_CUDA_RETURN_ON_FAIL(
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        SLANG_CUDA_RETURN_ON_FAIL(
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != CU_COMPUTEMODE_PROHIBITED)
        {
            if (major == 9999 && minor == 9999)
            {
                smPerMultiproc = 1;
            }
            else
            {
                smPerMultiproc = _calcSMCountPerMultiProcessor(major, minor);
            }

            int multiProcessorCount = 0, clockRate = 0;
            SLANG_CUDA_RETURN_ON_FAIL(cuDeviceGetAttribute(
                &multiProcessorCount,
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device));
            SLANG_CUDA_RETURN_ON_FAIL(
                cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
            uint64_t compute_perf = uint64_t(multiProcessorCount) * smPerMultiproc * clockRate;

            if (compute_perf > maxComputePerf)
            {
                maxComputePerf = compute_perf;
                maxPerfDevice = currentDevice;
            }
        }
        else
        {
            devicesProhibited++;
        }
    }

    if (maxPerfDevice < 0)
    {
        return SLANG_FAIL;
    }

    *outDeviceIndex = maxPerfDevice;
    return SLANG_OK;
}

SlangResult DeviceImpl::_initCuda(CUDAReportStyle reportType)
{
    static CUresult res = cuInit(0);
    SLANG_CUDA_RETURN_WITH_REPORT_ON_FAIL(res, reportType);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::getNativeDeviceHandles(InteropHandles* outHandles)
{
    outHandles->handles[0].handleValue = (uint64_t)m_device;
    outHandles->handles[0].api = InteropHandleAPI::CUDA;
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL DeviceImpl::initialize(const Desc& desc)
{
    SLANG_RETURN_ON_FAIL(slangContext.initialize(
        desc.slang,
        desc.extendedDescCount,
        desc.extendedDescs,
        SLANG_PTX,
        "sm_5_1",
        makeArray(slang::PreprocessorMacroDesc{"__CUDA_COMPUTE__", "1"}).getView()));

    SLANG_RETURN_ON_FAIL(RendererBase::initialize(desc));

    SLANG_RETURN_ON_FAIL(_initCuda(reportType));

    if (desc.adapterLUID)
    {
        int deviceCount = -1;
        cuDeviceGetCount(&deviceCount);
        for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
        {
            if (cuda::getAdapterLUID(deviceIndex) == *desc.adapterLUID)
            {
                m_deviceIndex = deviceIndex;
                break;
            }
        }
        if (m_deviceIndex >= deviceCount)
            return SLANG_E_INVALID_ARG;
    }
    else
    {
        SLANG_RETURN_ON_FAIL(_findMaxFlopsDeviceIndex(&m_deviceIndex));
    }

    m_context = new CUDAContext();

    SLANG_CUDA_RETURN_ON_FAIL(cuDeviceGet(&m_device, m_deviceIndex));

    SLANG_CUDA_RETURN_WITH_REPORT_ON_FAIL(
        cuCtxCreate(&m_context->m_context, 0, m_device),
        reportType);

    {
        // Not clear how to detect half support on CUDA. For now we'll assume we have it
        m_features.add("half");

        // CUDA has support for realtime clock
        m_features.add("realtime-clock");

        // Allows use of a ptr like type
        m_features.add("has-ptr");
    }

    // Initialize DeviceInfo
    {
        m_info.deviceType = DeviceType::CUDA;
        m_info.bindingStyle = BindingStyle::CUDA;
        m_info.projectionStyle = ProjectionStyle::DirectX;
        m_info.apiName = "CUDA";
        static const float kIdentity[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        ::memcpy(m_info.identityProjectionMatrix, kIdentity, sizeof(kIdentity));
        char deviceName[256];
        cuDeviceGetName(deviceName, sizeof(deviceName), m_device);
        m_adapterName = deviceName;
        m_info.adapterName = m_adapterName.begin();
        m_info.timestampFrequency = 1000000;
    }

    // Get device limits.
    {
        CUresult lastResult = CUDA_SUCCESS;
        auto getAttribute = [&](CUdevice_attribute attribute) -> int
        {
            int value;
            CUresult result = cuDeviceGetAttribute(&value, attribute, m_device);
            if (result != CUDA_SUCCESS)
                lastResult = result;
            return value;
        };

        DeviceLimits limits = {};

        limits.maxTextureDimension1D = getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH);
        limits.maxTextureDimension2D = Math::Min(
            getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH),
            getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT));
        limits.maxTextureDimension3D = Math::Min(
            getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH),
            Math::Min(
                getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT),
                getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)));
        limits.maxTextureDimensionCube =
            getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH);
        limits.maxTextureArrayLayers = Math::Min(
            getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS),
            getAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS));

        // limits.maxVertexInputElements
        // limits.maxVertexInputElementOffset
        // limits.maxVertexStreams
        // limits.maxVertexStreamStride

        limits.maxComputeThreadsPerGroup = getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        limits.maxComputeThreadGroupSize[0] = getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        limits.maxComputeThreadGroupSize[1] = getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        limits.maxComputeThreadGroupSize[2] = getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        limits.maxComputeDispatchThreadGroups[0] = getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        limits.maxComputeDispatchThreadGroups[1] = getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        limits.maxComputeDispatchThreadGroups[2] = getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);

        // limits.maxViewports
        // limits.maxViewportDimensions
        // limits.maxFramebufferDimensions

        // limits.maxShaderVisibleSamplers

        m_info.limits = limits;

        SLANG_CUDA_RETURN_ON_FAIL(lastResult);
    }

    return SLANG_OK;
}

Result DeviceImpl::getCUDAFormat(Format format, CUarray_format* outFormat)
{
    // TODO: Expand to cover all available formats that can be supported in CUDA
    switch (format)
    {
    case Format::R32G32B32A32_FLOAT:
    case Format::R32G32B32_FLOAT:
    case Format::R32G32_FLOAT:
    case Format::R32_FLOAT:
    case Format::D32_FLOAT:
        *outFormat = CU_AD_FORMAT_FLOAT;
        return SLANG_OK;
    case Format::R16G16B16A16_FLOAT:
    case Format::R16G16_FLOAT:
    case Format::R16_FLOAT:
        *outFormat = CU_AD_FORMAT_HALF;
        return SLANG_OK;
    case Format::R32G32B32A32_UINT:
    case Format::R32G32B32_UINT:
    case Format::R32G32_UINT:
    case Format::R32_UINT:
        *outFormat = CU_AD_FORMAT_UNSIGNED_INT32;
        return SLANG_OK;
    case Format::R16G16B16A16_UINT:
    case Format::R16G16_UINT:
    case Format::R16_UINT:
        *outFormat = CU_AD_FORMAT_UNSIGNED_INT16;
        return SLANG_OK;
    case Format::R8G8B8A8_UINT:
    case Format::R8G8_UINT:
    case Format::R8_UINT:
    case Format::R8G8B8A8_UNORM:
        *outFormat = CU_AD_FORMAT_UNSIGNED_INT8;
        return SLANG_OK;
    case Format::R32G32B32A32_SINT:
    case Format::R32G32B32_SINT:
    case Format::R32G32_SINT:
    case Format::R32_SINT:
        *outFormat = CU_AD_FORMAT_SIGNED_INT32;
        return SLANG_OK;
    case Format::R16G16B16A16_SINT:
    case Format::R16G16_SINT:
    case Format::R16_SINT:
        *outFormat = CU_AD_FORMAT_SIGNED_INT16;
        return SLANG_OK;
    case Format::R8G8B8A8_SINT:
    case Format::R8G8_SINT:
    case Format::R8_SINT:
        *outFormat = CU_AD_FORMAT_SIGNED_INT8;
        return SLANG_OK;
    default:
        SLANG_ASSERT(!"Only support R32_FLOAT/R8G8B8A8_UNORM formats for now");
        return SLANG_FAIL;
    }
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createTextureResource(
    const ITextureResource::Desc& desc,
    const ITextureResource::SubresourceData* initData,
    ITextureResource** outResource)
{
    TextureResource::Desc srcDesc = fixupTextureDesc(desc);

    RefPtr<TextureResourceImpl> tex = new TextureResourceImpl(srcDesc);
    tex->m_cudaContext = m_context;

    CUresourcetype resourceType;

    // The size of the element/texel in bytes
    size_t elementSize = 0;

    // Our `ITextureResource::Desc` uses an enumeration to specify
    // the "shape"/rank of a texture (1D, 2D, 3D, Cube), but CUDA's
    // `cuMipmappedArrayCreate` seemingly relies on a policy where
    // the extents of the array in dimenions above the rank are
    // specified as zero (e.g., a 1D texture requires `height==0`).
    //
    // We will start by massaging the extents as specified by the
    // user into a form that CUDA wants/expects, based on the
    // texture shape as specified in the `desc`.
    //
    int width = desc.size.width;
    int height = desc.size.height;
    int depth = desc.size.depth;
    switch (desc.type)
    {
    case IResource::Type::Texture1D:
        height = 0;
        depth = 0;
        break;

    case IResource::Type::Texture2D:
        depth = 0;
        break;

    case IResource::Type::Texture3D:
        break;

    case IResource::Type::TextureCube:
        depth = 1;
        break;
    }

    {
        CUarray_format format = CU_AD_FORMAT_FLOAT;
        int numChannels = 0;

        SLANG_RETURN_ON_FAIL(getCUDAFormat(desc.format, &format));
        FormatInfo info;
        gfxGetFormatInfo(desc.format, &info);
        numChannels = info.channelCount;

        switch (format)
        {
        case CU_AD_FORMAT_FLOAT:
            {
                elementSize = sizeof(float) * numChannels;
                break;
            }
        case CU_AD_FORMAT_HALF:
            {
                elementSize = sizeof(uint16_t) * numChannels;
                break;
            }
        case CU_AD_FORMAT_UNSIGNED_INT8:
            {
                elementSize = sizeof(uint8_t) * numChannels;
                break;
            }
        default:
            {
                SLANG_ASSERT(!"Only support R32_FLOAT/R8G8B8A8_UNORM formats for now");
                return SLANG_FAIL;
            }
        }

        if (desc.numMipLevels > 1)
        {
            resourceType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;

            CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
            memset(&arrayDesc, 0, sizeof(arrayDesc));

            arrayDesc.Width = width;
            arrayDesc.Height = height;
            arrayDesc.Depth = depth;
            arrayDesc.Format = format;
            arrayDesc.NumChannels = numChannels;
            arrayDesc.Flags = 0;

            if (desc.arraySize > 1)
            {
                if (desc.type == IResource::Type::Texture1D ||
                    desc.type == IResource::Type::Texture2D ||
                    desc.type == IResource::Type::TextureCube)
                {
                    arrayDesc.Flags |= CUDA_ARRAY3D_LAYERED;
                    arrayDesc.Depth = desc.arraySize;
                }
                else
                {
                    SLANG_ASSERT(!"Arrays only supported for 1D and 2D");
                    return SLANG_FAIL;
                }
            }

            if (desc.type == IResource::Type::TextureCube)
            {
                arrayDesc.Flags |= CUDA_ARRAY3D_CUBEMAP;
                arrayDesc.Depth *= 6;
            }

            SLANG_CUDA_RETURN_ON_FAIL(
                cuMipmappedArrayCreate(&tex->m_cudaMipMappedArray, &arrayDesc, desc.numMipLevels));
        }
        else
        {
            resourceType = CU_RESOURCE_TYPE_ARRAY;

            if (desc.arraySize > 1)
            {
                if (desc.type == IResource::Type::Texture1D ||
                    desc.type == IResource::Type::Texture2D ||
                    desc.type == IResource::Type::TextureCube)
                {
                    SLANG_ASSERT(!"Only 1D, 2D and Cube arrays supported");
                    return SLANG_FAIL;
                }

                CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
                memset(&arrayDesc, 0, sizeof(arrayDesc));

                // Set the depth as the array length
                arrayDesc.Depth = desc.arraySize;
                if (desc.type == IResource::Type::TextureCube)
                {
                    arrayDesc.Depth *= 6;
                }

                arrayDesc.Height = height;
                arrayDesc.Width = width;
                arrayDesc.Format = format;
                arrayDesc.NumChannels = numChannels;

                if (desc.type == IResource::Type::TextureCube)
                {
                    arrayDesc.Flags |= CUDA_ARRAY3D_CUBEMAP;
                }

                SLANG_CUDA_RETURN_ON_FAIL(cuArray3DCreate(&tex->m_cudaArray, &arrayDesc));
            }
            else if (
                desc.type == IResource::Type::Texture3D ||
                desc.type == IResource::Type::TextureCube)
            {
                CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
                memset(&arrayDesc, 0, sizeof(arrayDesc));

                arrayDesc.Depth = depth;
                arrayDesc.Height = height;
                arrayDesc.Width = width;
                arrayDesc.Format = format;
                arrayDesc.NumChannels = numChannels;

                arrayDesc.Flags = 0;

                // Handle cube texture
                if (desc.type == IResource::Type::TextureCube)
                {
                    arrayDesc.Depth = 6;
                    arrayDesc.Flags |= CUDA_ARRAY3D_CUBEMAP;
                }

                SLANG_CUDA_RETURN_ON_FAIL(cuArray3DCreate(&tex->m_cudaArray, &arrayDesc));
            }
            else
            {
                CUDA_ARRAY_DESCRIPTOR arrayDesc;
                memset(&arrayDesc, 0, sizeof(arrayDesc));

                arrayDesc.Height = height;
                arrayDesc.Width = width;
                arrayDesc.Format = format;
                arrayDesc.NumChannels = numChannels;

                // Allocate the array, will work for 1D or 2D case
                SLANG_CUDA_RETURN_ON_FAIL(cuArrayCreate(&tex->m_cudaArray, &arrayDesc));
            }
        }
    }

    // Work space for holding data for uploading if it needs to be rearranged
    if (initData)
    {
        List<uint8_t> workspace;
        for (int mipLevel = 0; mipLevel < desc.numMipLevels; ++mipLevel)
        {
            int mipWidth = width >> mipLevel;
            int mipHeight = height >> mipLevel;
            int mipDepth = depth >> mipLevel;

            mipWidth = (mipWidth == 0) ? 1 : mipWidth;
            mipHeight = (mipHeight == 0) ? 1 : mipHeight;
            mipDepth = (mipDepth == 0) ? 1 : mipDepth;

            // If it's a cubemap then the depth is always 6
            if (desc.type == IResource::Type::TextureCube)
            {
                mipDepth = 6;
            }

            auto dstArray = tex->m_cudaArray;
            if (tex->m_cudaMipMappedArray)
            {
                // Get the array for the mip level
                SLANG_CUDA_RETURN_ON_FAIL(
                    cuMipmappedArrayGetLevel(&dstArray, tex->m_cudaMipMappedArray, mipLevel));
            }
            SLANG_ASSERT(dstArray);

            // Check using the desc to see if it's plausible
            {
                CUDA_ARRAY_DESCRIPTOR arrayDesc;
                SLANG_CUDA_RETURN_ON_FAIL(cuArrayGetDescriptor(&arrayDesc, dstArray));

                SLANG_ASSERT(mipWidth == arrayDesc.Width);
                SLANG_ASSERT(
                    mipHeight == arrayDesc.Height || (mipHeight == 1 && arrayDesc.Height == 0));
            }

            const void* srcDataPtr = nullptr;

            if (desc.arraySize > 1)
            {
                SLANG_ASSERT(
                    desc.type == IResource::Type::Texture1D ||
                    desc.type == IResource::Type::Texture2D ||
                    desc.type == IResource::Type::TextureCube);

                // TODO(JS): Here I assume that arrays are just held contiguously within a
                // 'face' This seems reasonable and works with the Copy3D.
                const size_t faceSizeInBytes = elementSize * mipWidth * mipHeight;

                Index faceCount = desc.arraySize;
                if (desc.type == IResource::Type::TextureCube)
                {
                    faceCount *= 6;
                }

                const size_t mipSizeInBytes = faceSizeInBytes * faceCount;
                workspace.setCount(mipSizeInBytes);

                // We need to add the face data from each mip
                // We iterate over face count so we copy all of the cubemap faces
                for (Index j = 0; j < faceCount; j++)
                {
                    const auto srcData = initData[mipLevel + j * desc.numMipLevels].data;
                    // Copy over to the workspace to make contiguous
                    ::memcpy(workspace.begin() + faceSizeInBytes * j, srcData, faceSizeInBytes);
                }

                srcDataPtr = workspace.getBuffer();
            }
            else
            {
                if (desc.type == IResource::Type::TextureCube)
                {
                    size_t faceSizeInBytes = elementSize * mipWidth * mipHeight;

                    workspace.setCount(faceSizeInBytes * 6);
                    // Copy the data over to make contiguous
                    for (Index j = 0; j < 6; j++)
                    {
                        const auto srcData = initData[mipLevel + j * desc.numMipLevels].data;
                        ::memcpy(
                            workspace.getBuffer() + faceSizeInBytes * j,
                            srcData,
                            faceSizeInBytes);
                    }
                    srcDataPtr = workspace.getBuffer();
                }
                else
                {
                    const auto srcData = initData[mipLevel].data;
                    srcDataPtr = srcData;
                }
            }

            if (desc.arraySize > 1)
            {
                SLANG_ASSERT(
                    desc.type == IResource::Type::Texture1D ||
                    desc.type == IResource::Type::Texture2D ||
                    desc.type == IResource::Type::TextureCube);

                CUDA_MEMCPY3D copyParam;
                memset(&copyParam, 0, sizeof(copyParam));

                copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                copyParam.dstArray = dstArray;

                copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
                copyParam.srcHost = srcDataPtr;
                copyParam.srcPitch = mipWidth * elementSize;
                copyParam.WidthInBytes = copyParam.srcPitch;
                copyParam.Height = mipHeight;
                // Set the depth to the array length
                copyParam.Depth = desc.arraySize;

                if (desc.type == IResource::Type::TextureCube)
                {
                    copyParam.Depth *= 6;
                }

                SLANG_CUDA_RETURN_ON_FAIL(cuMemcpy3D(&copyParam));
            }
            else
            {
                switch (desc.type)
                {
                case IResource::Type::Texture1D:
                case IResource::Type::Texture2D:
                    {
                        CUDA_MEMCPY2D copyParam;
                        memset(&copyParam, 0, sizeof(copyParam));
                        copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                        copyParam.dstArray = dstArray;
                        copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
                        copyParam.srcHost = srcDataPtr;
                        copyParam.srcPitch = mipWidth * elementSize;
                        copyParam.WidthInBytes = copyParam.srcPitch;
                        copyParam.Height = mipHeight;
                        SLANG_CUDA_RETURN_ON_FAIL(cuMemcpy2D(&copyParam));
                        break;
                    }
                case IResource::Type::Texture3D:
                case IResource::Type::TextureCube:
                    {
                        CUDA_MEMCPY3D copyParam;
                        memset(&copyParam, 0, sizeof(copyParam));

                        copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                        copyParam.dstArray = dstArray;

                        copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
                        copyParam.srcHost = srcDataPtr;
                        copyParam.srcPitch = mipWidth * elementSize;
                        copyParam.WidthInBytes = copyParam.srcPitch;
                        copyParam.Height = mipHeight;
                        copyParam.Depth = mipDepth;

                        SLANG_CUDA_RETURN_ON_FAIL(cuMemcpy3D(&copyParam));
                        break;
                    }

                default:
                    {
                        SLANG_ASSERT(!"Not implemented");
                        break;
                    }
                }
            }
        }
    }
    // Set up texture sampling parameters, and create final texture obj

    {
        CUDA_RESOURCE_DESC resDesc;
        memset(&resDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        resDesc.resType = resourceType;

        if (tex->m_cudaArray)
        {
            resDesc.res.array.hArray = tex->m_cudaArray;
        }
        if (tex->m_cudaMipMappedArray)
        {
            resDesc.res.mipmap.hMipmappedArray = tex->m_cudaMipMappedArray;
        }

        // If the texture might be used as a UAV, then we need to allocate
        // a CUDA "surface" for it.
        //
        // Note: We cannot do this unconditionally, because it will fail
        // on surfaces that are not usable as UAVs (e.g., those with
        // mipmaps).
        //
        // TODO: We should really only be allocating the array at the
        // time we create a resource, and then allocate the surface or
        // texture objects as part of view creation.
        //
        if (desc.allowedStates.contains(ResourceState::UnorderedAccess))
        {
            // On CUDA surfaces only support a single MIP map
            SLANG_ASSERT(desc.numMipLevels == 1);

            SLANG_CUDA_RETURN_ON_FAIL(cuSurfObjectCreate(&tex->m_cudaSurfObj, &resDesc));
        }


        // Create handle for sampling.
        CUDA_TEXTURE_DESC texDesc;
        memset(&texDesc, 0, sizeof(CUDA_TEXTURE_DESC));
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
        texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
        texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
        texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;

        SLANG_CUDA_RETURN_ON_FAIL(
            cuTexObjectCreate(&tex->m_cudaTexObj, &resDesc, &texDesc, nullptr));
    }

    returnComPtr(outResource, tex);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createBufferResource(
    const IBufferResource::Desc& descIn,
    const void* initData,
    IBufferResource** outResource)
{
    auto desc = fixupBufferDesc(descIn);
    RefPtr<BufferResourceImpl> resource = new BufferResourceImpl(desc);
    resource->m_cudaContext = m_context;
    SLANG_CUDA_RETURN_ON_FAIL(cuMemAllocManaged(
        (CUdeviceptr*)(&resource->m_cudaMemory),
        desc.sizeInBytes,
        CU_MEM_ATTACH_GLOBAL));
    if (initData)
    {
        SLANG_CUDA_RETURN_ON_FAIL(
            cuMemcpy((CUdeviceptr)resource->m_cudaMemory, (CUdeviceptr)initData, desc.sizeInBytes));
    }
    returnComPtr(outResource, resource);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createBufferFromSharedHandle(
    InteropHandle handle,
    const IBufferResource::Desc& desc,
    IBufferResource** outResource)
{
    if (handle.handleValue == 0)
    {
        *outResource = nullptr;
        return SLANG_OK;
    }

    RefPtr<BufferResourceImpl> resource = new BufferResourceImpl(desc);
    resource->m_cudaContext = m_context;

    // CUDA manages sharing of buffers through the idea of an
    // "external memory" object, which represents the relationship
    // with another API's objects. In order to create this external
    // memory association, we first need to fill in a descriptor struct.
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
    switch (handle.api)
    {
    case InteropHandleAPI::D3D12:
        externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        break;
    case InteropHandleAPI::Vulkan:
        externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
        break;
    default:
        return SLANG_FAIL;
    }
    externalMemoryHandleDesc.handle.win32.handle = (void*)handle.handleValue;
    externalMemoryHandleDesc.size = desc.sizeInBytes;
    externalMemoryHandleDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;

    // Once we have filled in the descriptor, we can request
    // that CUDA create the required association between the
    // external buffer and its own memory.
    CUexternalMemory externalMemory;
    SLANG_CUDA_RETURN_ON_FAIL(cuImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
    resource->m_cudaExternalMemory = externalMemory;

    // The CUDA "external memory" handle is not itself a device
    // pointer, so we need to query for a suitable device address
    // for the buffer with another call.
    //
    // Just as for the external memory, we fill in a descriptor
    // structure (although in this case we only need to specify
    // the size).
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.size = desc.sizeInBytes;

    // Finally, we can "map" the buffer to get a device address.
    void* deviceAddress;
    SLANG_CUDA_RETURN_ON_FAIL(
        cuExternalMemoryGetMappedBuffer((CUdeviceptr*)&deviceAddress, externalMemory, &bufferDesc));
    resource->m_cudaMemory = deviceAddress;

    returnComPtr(outResource, resource);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createTextureFromSharedHandle(
    InteropHandle handle,
    const ITextureResource::Desc& desc,
    const size_t size,
    ITextureResource** outResource)
{
    if (handle.handleValue == 0)
    {
        *outResource = nullptr;
        return SLANG_OK;
    }

    RefPtr<TextureResourceImpl> resource = new TextureResourceImpl(desc);
    resource->m_cudaContext = m_context;

    // CUDA manages sharing of buffers through the idea of an
    // "external memory" object, which represents the relationship
    // with another API's objects. In order to create this external
    // memory association, we first need to fill in a descriptor struct.
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
    switch (handle.api)
    {
    case InteropHandleAPI::D3D12:
        externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        break;
    case InteropHandleAPI::Vulkan:
        externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
        break;
    default:
        return SLANG_FAIL;
    }
    externalMemoryHandleDesc.handle.win32.handle = (void*)handle.handleValue;
    externalMemoryHandleDesc.size = size;
    externalMemoryHandleDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;

    CUexternalMemory externalMemory;
    SLANG_CUDA_RETURN_ON_FAIL(cuImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
    resource->m_cudaExternalMemory = externalMemory;

    FormatInfo formatInfo;
    SLANG_RETURN_ON_FAIL(gfxGetFormatInfo(desc.format, &formatInfo));
    CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
    arrayDesc.Depth = desc.size.depth;
    arrayDesc.Height = desc.size.height;
    arrayDesc.Width = desc.size.width;
    arrayDesc.NumChannels = formatInfo.channelCount;
    getCUDAFormat(desc.format, &arrayDesc.Format);
    arrayDesc.Flags = 0; // TODO: Flags? CUDA_ARRAY_LAYERED/SURFACE_LDST/CUBEMAP/TEXTURE_GATHER

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC externalMemoryMipDesc;
    memset(&externalMemoryMipDesc, 0, sizeof(externalMemoryMipDesc));
    externalMemoryMipDesc.offset = 0;
    externalMemoryMipDesc.arrayDesc = arrayDesc;
    externalMemoryMipDesc.numLevels = desc.numMipLevels;

    CUmipmappedArray mipArray;
    SLANG_CUDA_RETURN_ON_FAIL(
        cuExternalMemoryGetMappedMipmappedArray(&mipArray, externalMemory, &externalMemoryMipDesc));
    resource->m_cudaMipMappedArray = mipArray;

    CUarray cuArray;
    SLANG_CUDA_RETURN_ON_FAIL(cuMipmappedArrayGetLevel(&cuArray, mipArray, 0));
    resource->m_cudaArray = cuArray;

    CUDA_RESOURCE_DESC surfDesc;
    memset(&surfDesc, 0, sizeof(surfDesc));
    surfDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    surfDesc.res.array.hArray = cuArray;

    CUsurfObject surface;
    SLANG_CUDA_RETURN_ON_FAIL(cuSurfObjectCreate(&surface, &surfDesc));
    resource->m_cudaSurfObj = surface;

    returnComPtr(outResource, resource);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createTextureView(
    ITextureResource* texture,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    RefPtr<ResourceViewImpl> view = new ResourceViewImpl();
    view->m_desc = desc;
    view->textureResource = dynamic_cast<TextureResourceImpl*>(texture);
    returnComPtr(outView, view);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createBufferView(
    IBufferResource* buffer,
    IBufferResource* counterBuffer,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    RefPtr<ResourceViewImpl> view = new ResourceViewImpl();
    view->m_desc = desc;
    view->memoryResource = dynamic_cast<BufferResourceImpl*>(buffer);
    returnComPtr(outView, view);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
DeviceImpl::createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outPool)
{
    RefPtr<QueryPoolImpl> pool = new QueryPoolImpl();
    SLANG_RETURN_ON_FAIL(pool->init(desc));
    returnComPtr(outPool, pool);
    return SLANG_OK;
}

Result DeviceImpl::createShaderObjectLayout(
    slang::ISession* session,
    slang::TypeLayoutReflection* typeLayout,
    ShaderObjectLayoutBase** outLayout)
{
    RefPtr<ShaderObjectLayoutImpl> cudaLayout;
    cudaLayout = new ShaderObjectLayoutImpl(this, session, typeLayout);
    returnRefPtrMove(outLayout, cudaLayout);
    return SLANG_OK;
}

Result DeviceImpl::createShaderObject(ShaderObjectLayoutBase* layout, IShaderObject** outObject)
{
    RefPtr<ShaderObjectImpl> result = new ShaderObjectImpl();
    SLANG_RETURN_ON_FAIL(result->init(this, dynamic_cast<ShaderObjectLayoutImpl*>(layout)));
    returnComPtr(outObject, result);
    return SLANG_OK;
}

Result DeviceImpl::createMutableShaderObject(
    ShaderObjectLayoutBase* layout,
    IShaderObject** outObject)
{
    RefPtr<MutableShaderObjectImpl> result = new MutableShaderObjectImpl();
    SLANG_RETURN_ON_FAIL(result->init(this, dynamic_cast<ShaderObjectLayoutImpl*>(layout)));
    returnComPtr(outObject, result);
    return SLANG_OK;
}

Result DeviceImpl::createRootShaderObject(IShaderProgram* program, ShaderObjectBase** outObject)
{
    auto cudaProgram = dynamic_cast<ShaderProgramImpl*>(program);
    auto cudaLayout = cudaProgram->layout;

    RefPtr<RootShaderObjectImpl> result = new RootShaderObjectImpl();
    SLANG_RETURN_ON_FAIL(result->init(this, cudaLayout));
    returnRefPtrMove(outObject, result);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createProgram(
    const IShaderProgram::Desc& desc,
    IShaderProgram** outProgram,
    ISlangBlob** outDiagnosticBlob)
{
    // If this is a specializable program, we just keep a reference to the slang program and
    // don't actually create any kernels. This program will be specialized later when we know
    // the shader object bindings.
    RefPtr<ShaderProgramImpl> cudaProgram = new ShaderProgramImpl();
    cudaProgram->init(desc);
    cudaProgram->cudaContext = m_context;
    if (desc.slangGlobalScope->getSpecializationParamCount() != 0)
    {
        cudaProgram->layout =
            new RootShaderObjectLayoutImpl(this, desc.slangGlobalScope->getLayout());
        returnComPtr(outProgram, cudaProgram);
        return SLANG_OK;
    }

    ComPtr<ISlangBlob> kernelCode;
    ComPtr<ISlangBlob> diagnostics;
    auto compileResult = getEntryPointCodeFromShaderCache(
        desc.slangGlobalScope,
        (SlangInt)0,
        0,
        kernelCode.writeRef(),
        diagnostics.writeRef());
    if (diagnostics)
    {
        getDebugCallback()->handleMessage(
            compileResult == SLANG_OK ? DebugMessageType::Warning : DebugMessageType::Error,
            DebugMessageSource::Slang,
            (char*)diagnostics->getBufferPointer());
        if (outDiagnosticBlob)
            returnComPtr(outDiagnosticBlob, diagnostics);
    }
    SLANG_RETURN_ON_FAIL(compileResult);

    SLANG_CUDA_RETURN_ON_FAIL(
        cuModuleLoadData(&cudaProgram->cudaModule, kernelCode->getBufferPointer()));
    cudaProgram->kernelName =
        desc.slangGlobalScope->getLayout()->getEntryPointByIndex(0)->getName();
    SLANG_CUDA_RETURN_ON_FAIL(cuModuleGetFunction(
        &cudaProgram->cudaKernel,
        cudaProgram->cudaModule,
        cudaProgram->kernelName.getBuffer()));

    auto slangGlobalScope = desc.slangGlobalScope;
    if (slangGlobalScope)
    {
        cudaProgram->slangGlobalScope = slangGlobalScope;

        auto slangProgramLayout = slangGlobalScope->getLayout();
        if (!slangProgramLayout)
            return SLANG_FAIL;

        RefPtr<RootShaderObjectLayoutImpl> cudaLayout;
        cudaLayout = new RootShaderObjectLayoutImpl(this, slangProgramLayout);
        cudaLayout->programLayout = slangProgramLayout;
        cudaProgram->layout = cudaLayout;
    }

    returnComPtr(outProgram, cudaProgram);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createComputePipelineState(
    const ComputePipelineStateDesc& desc,
    IPipelineState** outState)
{
    RefPtr<ComputePipelineStateImpl> state = new ComputePipelineStateImpl();
    state->shaderProgram = static_cast<ShaderProgramImpl*>(desc.program);
    state->init(desc);
    returnComPtr(outState, state);
    return Result();
}

void* DeviceImpl::map(IBufferResource* buffer)
{
    return static_cast<BufferResourceImpl*>(buffer)->m_cudaMemory;
}

void DeviceImpl::unmap(IBufferResource* buffer)
{
    SLANG_UNUSED(buffer);
}

SLANG_NO_THROW const DeviceInfo& SLANG_MCALL DeviceImpl::getDeviceInfo() const
{
    return m_info;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createTransientResourceHeap(
    const ITransientResourceHeap::Desc& desc,
    ITransientResourceHeap** outHeap)
{
    RefPtr<TransientResourceHeapImpl> result = new TransientResourceHeapImpl();
    SLANG_RETURN_ON_FAIL(result->init(this, desc));
    returnComPtr(outHeap, result);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
DeviceImpl::createCommandQueue(const ICommandQueue::Desc& desc, ICommandQueue** outQueue)
{
    RefPtr<CommandQueueImpl> queue = new CommandQueueImpl();
    queue->init(this);
    returnComPtr(outQueue, queue);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createSwapchain(
    const ISwapchain::Desc& desc,
    WindowHandle window,
    ISwapchain** outSwapchain)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(window);
    SLANG_UNUSED(outSwapchain);
    return SLANG_FAIL;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createFramebufferLayout(
    const IFramebufferLayout::Desc& desc,
    IFramebufferLayout** outLayout)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outLayout);
    return SLANG_FAIL;
}

SLANG_NO_THROW Result SLANG_MCALL
DeviceImpl::createFramebuffer(const IFramebuffer::Desc& desc, IFramebuffer** outFramebuffer)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outFramebuffer);
    return SLANG_FAIL;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createRenderPassLayout(
    const IRenderPassLayout::Desc& desc,
    IRenderPassLayout** outRenderPassLayout)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outRenderPassLayout);
    return SLANG_FAIL;
}

SLANG_NO_THROW Result SLANG_MCALL
DeviceImpl::createSamplerState(ISamplerState::Desc const& desc, ISamplerState** outSampler)
{
    SLANG_UNUSED(desc);
    *outSampler = nullptr;
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
DeviceImpl::createInputLayout(IInputLayout::Desc const& desc, IInputLayout** outLayout)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outLayout);
    return SLANG_E_NOT_AVAILABLE;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createGraphicsPipelineState(
    const GraphicsPipelineStateDesc& desc,
    IPipelineState** outState)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outState);
    return SLANG_E_NOT_AVAILABLE;
}

SLANG_NO_THROW SlangResult SLANG_MCALL DeviceImpl::readTextureResource(
    ITextureResource* texture,
    ResourceState state,
    ISlangBlob** outBlob,
    size_t* outRowPitch,
    size_t* outPixelSize)
{
    auto textureImpl = static_cast<TextureResourceImpl*>(texture);

    List<uint8_t> blobData;

    auto desc = textureImpl->getDesc();
    auto width = desc->size.width;
    auto height = desc->size.height;
    FormatInfo sizeInfo;
    SLANG_RETURN_ON_FAIL(gfxGetFormatInfo(desc->format, &sizeInfo));
    size_t pixelSize = sizeInfo.blockSizeInBytes / sizeInfo.pixelsPerBlock;
    size_t rowPitch = width * pixelSize;
    size_t size = height * rowPitch;
    blobData.setCount((Index)size);

    CUDA_MEMCPY2D copyParam;
    memset(&copyParam, 0, sizeof(copyParam));

    copyParam.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParam.srcArray = textureImpl->m_cudaArray;

    copyParam.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyParam.dstHost = blobData.getBuffer();
    copyParam.dstPitch = rowPitch;
    copyParam.WidthInBytes = copyParam.dstPitch;
    copyParam.Height = height;
    SLANG_CUDA_RETURN_ON_FAIL(cuMemcpy2D(&copyParam));

    *outRowPitch = rowPitch;
    *outPixelSize = pixelSize;

    auto blob = ListBlob::moveCreate(blobData);

    returnComPtr(outBlob, blob);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::readBufferResource(
    IBufferResource* buffer,
    size_t offset,
    size_t size,
    ISlangBlob** outBlob)
{
    auto bufferImpl = static_cast<BufferResourceImpl*>(buffer);

    List<uint8_t> blobData;

    blobData.setCount((Index)size);
    cuMemcpy(
        (CUdeviceptr)blobData.getBuffer(),
        (CUdeviceptr)((uint8_t*)bufferImpl->m_cudaMemory + offset),
        size);

    auto blob = ListBlob::moveCreate(blobData);

    returnComPtr(outBlob, blob);
    return SLANG_OK;
}

} // namespace cuda
#endif
} // namespace gfx
