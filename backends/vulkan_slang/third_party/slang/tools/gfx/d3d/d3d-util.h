// d3d-util.h
#pragma once

#include "../flag-combiner.h"
#include "core/slang-basic.h"
#include "core/slang-platform.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-gfx.h"

#include <d3d12.h>
#include <d3dcommon.h>
#include <dxgi.h>
#include <dxgiformat.h>
#include <stdint.h>

#if defined(__ID3D12Device5_FWD_DEFINED__) && defined(__ID3D12GraphicsCommandList4_FWD_DEFINED__)
#define SLANG_GFX_HAS_DXR_SUPPORT 1
#else
#define SLANG_GFX_HAS_DXR_SUPPORT 0
typedef ISlangUnknown ID3D12Device5;
typedef ISlangUnknown ID3D12GraphicsCommandList4;

#endif

namespace gfx
{

class D3DUtil
{
public:
    enum UsageType
    {
        USAGE_UNKNOWN,       ///< Generally used to mark an error
        USAGE_TARGET,        ///< Format should be used when written as target
        USAGE_DEPTH_STENCIL, ///< Format should be used when written as depth stencil
        USAGE_SRV,           ///< Format if being read as srv
        USAGE_COUNT_OF,
    };
    enum UsageFlag
    {
        USAGE_FLAG_MULTI_SAMPLE = 0x1, ///< If set will be used form multi sampling (such as MSAA)
        USAGE_FLAG_SRV = 0x2, ///< If set means will be used as a shader resource view (SRV)
    };

    /// Get primitive topology as D3D primitive topology
    static D3D_PRIMITIVE_TOPOLOGY getPrimitiveTopology(PrimitiveTopology prim);

    static D3D12_PRIMITIVE_TOPOLOGY_TYPE getPrimitiveType(PrimitiveType type);

    static D3D12_PRIMITIVE_TOPOLOGY_TYPE getPrimitiveType(PrimitiveTopology topology);

    static D3D12_COMPARISON_FUNC getComparisonFunc(ComparisonFunc func);

    static D3D12_DEPTH_STENCILOP_DESC translateStencilOpDesc(DepthStencilOpDesc desc);

    /// Calculate size taking into account alignment. Alignment must be a power of 2
    static UInt calcAligned(UInt size, UInt alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /// Compile HLSL code to DXBC
    static Slang::Result compileHLSLShader(
        char const* sourcePath,
        char const* source,
        char const* entryPointName,
        char const* dxProfileName,
        Slang::ComPtr<ID3DBlob>& shaderBlobOut);

    /// Given a slang pixel format returns the equivalent DXGI_ pixel format. If the format is not
    /// known, will return DXGI_FORMAT_UNKNOWN
    static DXGI_FORMAT getMapFormat(Format format);

    /// Given the usage, flags, and format will return the most suitable format. Will return
    /// DXGI_UNKNOWN if combination is not possible
    static DXGI_FORMAT calcFormat(UsageType usage, DXGI_FORMAT format);
    /// Calculate appropriate format for creating a buffer for usage and flags
    static DXGI_FORMAT calcResourceFormat(UsageType usage, Int usageFlags, DXGI_FORMAT format);
    /// True if the type is 'typeless'
    static bool isTypeless(DXGI_FORMAT format);

    /// Returns number of bits used for color channel for format (for channels with multiple sizes,
    /// returns smallest ie RGB565 -> 5)
    static Int getNumColorChannelBits(DXGI_FORMAT fmt);

    static SlangResult createFactory(
        DeviceCheckFlags flags,
        Slang::ComPtr<IDXGIFactory>& outFactory);

    /// Get the dxgiModule
    static Slang::SharedLibrary::Handle getDxgiModule();

    /// Find adapters
    static SlangResult findAdapters(
        DeviceCheckFlags flags,
        const AdapterLUID* adapterLUID,
        IDXGIFactory* dxgiFactory,
        Slang::List<Slang::ComPtr<IDXGIAdapter>>& dxgiAdapters);
    /// Find adapters
    static SlangResult findAdapters(
        DeviceCheckFlags flags,
        const AdapterLUID* adapterLUID,
        Slang::List<Slang::ComPtr<IDXGIAdapter>>& dxgiAdapters);

    static AdapterLUID getAdapterLUID(IDXGIAdapter* dxgiAdapter);

    /// True if the adapter is warp
    static bool isWarp(IDXGIFactory* dxgiFactory, IDXGIAdapter* adapter);

    static bool isUAVBinding(slang::BindingType bindingType);

    static int getShaderModelFromProfileName(const char* profile);

    static uint32_t getPlaneSlice(DXGI_FORMAT format, TextureAspect aspect);

    static uint32_t getPlaneSliceCount(DXGI_FORMAT format);

    static D3D12_INPUT_CLASSIFICATION getInputSlotClass(InputSlotClass slotClass);

    static D3D12_FILL_MODE getFillMode(FillMode mode);

    static D3D12_CULL_MODE getCullMode(CullMode mode);

    static D3D12_BLEND_OP getBlendOp(BlendOp op);

    static D3D12_BLEND getBlendFactor(BlendFactor factor);

    static uint32_t getSubresourceIndex(
        uint32_t mipIndex,
        uint32_t arrayIndex,
        uint32_t planeIndex,
        uint32_t mipLevelCount,
        uint32_t arraySize);

    static uint32_t getSubresourceMipLevel(uint32_t subresourceIndex, uint32_t mipLevelCount);

    static D3D12_RESOURCE_STATES getResourceState(ResourceState state);

    static SlangResult reportLiveObjects();

    /// Call after a DXGI_ERROR_DEVICE_REMOVED/DXGI_ERROR_DEVICE_RESET on present, to wait for
    /// dumping to complete. Will return SLANG_OK if wait happened successfully
    static SlangResult waitForCrashDumpCompletion(HRESULT res);
};

#if SLANG_GFX_HAS_DXR_SUPPORT
struct D3DAccelerationStructureInputsBuilder
{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS desc = {};
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    Slang::List<D3D12_RAYTRACING_GEOMETRY_DESC> geomDescs;
    Slang::Result build(
        const IAccelerationStructure::BuildInputs& buildInputs,
        IDebugCallback* callback);

private:
    D3D12_RAYTRACING_GEOMETRY_FLAGS
    translateGeometryFlags(IAccelerationStructure::GeometryFlags::Enum flags)
    {
        return (D3D12_RAYTRACING_GEOMETRY_FLAGS)flags;
    }
};
#endif
} // namespace gfx
