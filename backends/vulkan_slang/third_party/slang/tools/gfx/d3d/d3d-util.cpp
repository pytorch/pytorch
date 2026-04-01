// d3d-util.cpp
#include "d3d-util.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <dxgidebug.h>
#if SLANG_ENABLE_DXBC_SUPPORT
#include <d3dcompiler.h>
#endif

// We will use the C standard library just for printing error messages.
#include "core/slang-basic.h"
#include "core/slang-platform.h"

#include <stdio.h>

#ifdef GFX_NV_AFTERMATH
#include "GFSDK_Aftermath.h"
#include "GFSDK_Aftermath_Defines.h"
#include "GFSDK_Aftermath_GpuCrashDump.h"
#include "core/slang-process.h"
#endif

namespace gfx
{
using namespace Slang;

/* static */ D3D_PRIMITIVE_TOPOLOGY D3DUtil::getPrimitiveTopology(PrimitiveTopology topology)
{
    switch (topology)
    {
    case PrimitiveTopology::TriangleList:
        return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    case PrimitiveTopology::TriangleStrip:
        return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    case PrimitiveTopology::LineList:
        return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    case PrimitiveTopology::LineStrip:
        return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case PrimitiveTopology::PointList:
        return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    default:
        break;
    }
    return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE D3DUtil::getPrimitiveType(PrimitiveTopology topology)
{
    switch (topology)
    {
    case PrimitiveTopology::TriangleList:
    case PrimitiveTopology::TriangleStrip:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    case PrimitiveTopology::LineList:
    case PrimitiveTopology::LineStrip:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    case PrimitiveTopology::PointList:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    default:
        break;
    }
    return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE D3DUtil::getPrimitiveType(PrimitiveType type)
{
    switch (type)
    {
    case PrimitiveType::Point:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    case PrimitiveType::Line:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    case PrimitiveType::Triangle:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    case PrimitiveType::Patch:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    default:
        break;
    }
    return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
}

D3D12_COMPARISON_FUNC D3DUtil::getComparisonFunc(ComparisonFunc func)
{
    switch (func)
    {
    case gfx::ComparisonFunc::Never:
        return D3D12_COMPARISON_FUNC_NEVER;
    case gfx::ComparisonFunc::Less:
        return D3D12_COMPARISON_FUNC_LESS;
    case gfx::ComparisonFunc::Equal:
        return D3D12_COMPARISON_FUNC_EQUAL;
    case gfx::ComparisonFunc::LessEqual:
        return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    case gfx::ComparisonFunc::Greater:
        return D3D12_COMPARISON_FUNC_GREATER;
    case gfx::ComparisonFunc::NotEqual:
        return D3D12_COMPARISON_FUNC_NOT_EQUAL;
    case gfx::ComparisonFunc::GreaterEqual:
        return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    case gfx::ComparisonFunc::Always:
        return D3D12_COMPARISON_FUNC_ALWAYS;
    default:
        return D3D12_COMPARISON_FUNC_NEVER;
    }
}

static D3D12_STENCIL_OP translateStencilOp(StencilOp op)
{
    switch (op)
    {
    case gfx::StencilOp::Keep:
        return D3D12_STENCIL_OP_KEEP;
    case gfx::StencilOp::Zero:
        return D3D12_STENCIL_OP_ZERO;
    case gfx::StencilOp::Replace:
        return D3D12_STENCIL_OP_REPLACE;
    case gfx::StencilOp::IncrementSaturate:
        return D3D12_STENCIL_OP_INCR_SAT;
    case gfx::StencilOp::DecrementSaturate:
        return D3D12_STENCIL_OP_DECR_SAT;
    case gfx::StencilOp::Invert:
        return D3D12_STENCIL_OP_INVERT;
    case gfx::StencilOp::IncrementWrap:
        return D3D12_STENCIL_OP_INCR;
    case gfx::StencilOp::DecrementWrap:
        return D3D12_STENCIL_OP_DECR;
    default:
        return D3D12_STENCIL_OP_KEEP;
    }
}

D3D12_DEPTH_STENCILOP_DESC D3DUtil::translateStencilOpDesc(DepthStencilOpDesc desc)
{
    D3D12_DEPTH_STENCILOP_DESC rs;
    rs.StencilDepthFailOp = translateStencilOp(desc.stencilDepthFailOp);
    rs.StencilFailOp = translateStencilOp(desc.stencilFailOp);
    rs.StencilFunc = getComparisonFunc(desc.stencilFunc);
    rs.StencilPassOp = translateStencilOp(desc.stencilPassOp);
    return rs;
}

/* static */ DXGI_FORMAT D3DUtil::getMapFormat(Format format)
{
    switch (format)
    {
    case Format::R32G32B32A32_TYPELESS:
        return DXGI_FORMAT_R32G32B32A32_TYPELESS;
    case Format::R32G32B32_TYPELESS:
        return DXGI_FORMAT_R32G32B32_TYPELESS;
    case Format::R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_TYPELESS;
    case Format::R32_TYPELESS:
        return DXGI_FORMAT_R32_TYPELESS;

    case Format::R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    case Format::R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_TYPELESS;
    case Format::R16_TYPELESS:
        return DXGI_FORMAT_R16_TYPELESS;

    case Format::R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case Format::R8G8_TYPELESS:
        return DXGI_FORMAT_R8G8_TYPELESS;
    case Format::R8_TYPELESS:
        return DXGI_FORMAT_R8_TYPELESS;
    case Format::B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;

    case Format::R32G32B32A32_FLOAT:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case Format::R32G32B32_FLOAT:
        return DXGI_FORMAT_R32G32B32_FLOAT;
    case Format::R32G32_FLOAT:
        return DXGI_FORMAT_R32G32_FLOAT;
    case Format::R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;

    case Format::R16G16B16A16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case Format::R16G16_FLOAT:
        return DXGI_FORMAT_R16G16_FLOAT;
    case Format::R16_FLOAT:
        return DXGI_FORMAT_R16_FLOAT;

    case Format::R64_UINT:
        return DXGI_FORMAT_R32G32_UINT;

    case Format::R32G32B32A32_UINT:
        return DXGI_FORMAT_R32G32B32A32_UINT;
    case Format::R32G32B32_UINT:
        return DXGI_FORMAT_R32G32B32_UINT;
    case Format::R32G32_UINT:
        return DXGI_FORMAT_R32G32_UINT;
    case Format::R32_UINT:
        return DXGI_FORMAT_R32_UINT;

    case Format::R16G16B16A16_UINT:
        return DXGI_FORMAT_R16G16B16A16_UINT;
    case Format::R16G16_UINT:
        return DXGI_FORMAT_R16G16_UINT;
    case Format::R16_UINT:
        return DXGI_FORMAT_R16_UINT;

    case Format::R8G8B8A8_UINT:
        return DXGI_FORMAT_R8G8B8A8_UINT;
    case Format::R8G8_UINT:
        return DXGI_FORMAT_R8G8_UINT;
    case Format::R8_UINT:
        return DXGI_FORMAT_R8_UINT;

    case Format::R64_SINT:
        return DXGI_FORMAT_R32G32_SINT;

    case Format::R32G32B32A32_SINT:
        return DXGI_FORMAT_R32G32B32A32_SINT;
    case Format::R32G32B32_SINT:
        return DXGI_FORMAT_R32G32B32_SINT;
    case Format::R32G32_SINT:
        return DXGI_FORMAT_R32G32_SINT;
    case Format::R32_SINT:
        return DXGI_FORMAT_R32_SINT;

    case Format::R16G16B16A16_SINT:
        return DXGI_FORMAT_R16G16B16A16_SINT;
    case Format::R16G16_SINT:
        return DXGI_FORMAT_R16G16_SINT;
    case Format::R16_SINT:
        return DXGI_FORMAT_R16_SINT;

    case Format::R8G8B8A8_SINT:
        return DXGI_FORMAT_R8G8B8A8_SINT;
    case Format::R8G8_SINT:
        return DXGI_FORMAT_R8G8_SINT;
    case Format::R8_SINT:
        return DXGI_FORMAT_R8_SINT;

    case Format::R16G16B16A16_UNORM:
        return DXGI_FORMAT_R16G16B16A16_UNORM;
    case Format::R16G16_UNORM:
        return DXGI_FORMAT_R16G16_UNORM;
    case Format::R16_UNORM:
        return DXGI_FORMAT_R16_UNORM;

    case Format::R8G8B8A8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case Format::R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case Format::R8G8_UNORM:
        return DXGI_FORMAT_R8G8_UNORM;
    case Format::R8_UNORM:
        return DXGI_FORMAT_R8_UNORM;
    case Format::B8G8R8A8_UNORM:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case Format::B8G8R8X8_UNORM:
        return DXGI_FORMAT_B8G8R8X8_UNORM;
    case Format::B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case Format::B8G8R8X8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;

    case Format::R16G16B16A16_SNORM:
        return DXGI_FORMAT_R16G16B16A16_SNORM;
    case Format::R16G16_SNORM:
        return DXGI_FORMAT_R16G16_SNORM;
    case Format::R16_SNORM:
        return DXGI_FORMAT_R16_SNORM;

    case Format::R8G8B8A8_SNORM:
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    case Format::R8G8_SNORM:
        return DXGI_FORMAT_R8G8_SNORM;
    case Format::R8_SNORM:
        return DXGI_FORMAT_R8_SNORM;

    case Format::D32_FLOAT:
        return DXGI_FORMAT_D32_FLOAT;
    case Format::D16_UNORM:
        return DXGI_FORMAT_D16_UNORM;
    case Format::D32_FLOAT_S8_UINT:
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case Format::R32_FLOAT_X32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;

    case Format::B4G4R4A4_UNORM:
        return DXGI_FORMAT_B4G4R4A4_UNORM;
    case Format::B5G6R5_UNORM:
        return DXGI_FORMAT_B5G6R5_UNORM;
    case Format::B5G5R5A1_UNORM:
        return DXGI_FORMAT_B5G5R5A1_UNORM;

    case Format::R9G9B9E5_SHAREDEXP:
        return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    case Format::R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    case Format::R10G10B10A2_UINT:
        return DXGI_FORMAT_R10G10B10A2_UINT;
    case Format::R10G10B10A2_UNORM:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case Format::R11G11B10_FLOAT:
        return DXGI_FORMAT_R11G11B10_FLOAT;

    case Format::BC1_UNORM:
        return DXGI_FORMAT_BC1_UNORM;
    case Format::BC1_UNORM_SRGB:
        return DXGI_FORMAT_BC1_UNORM_SRGB;
    case Format::BC2_UNORM:
        return DXGI_FORMAT_BC2_UNORM;
    case Format::BC2_UNORM_SRGB:
        return DXGI_FORMAT_BC2_UNORM_SRGB;
    case Format::BC3_UNORM:
        return DXGI_FORMAT_BC3_UNORM;
    case Format::BC3_UNORM_SRGB:
        return DXGI_FORMAT_BC3_UNORM_SRGB;
    case Format::BC4_UNORM:
        return DXGI_FORMAT_BC4_UNORM;
    case Format::BC4_SNORM:
        return DXGI_FORMAT_BC4_SNORM;
    case Format::BC5_UNORM:
        return DXGI_FORMAT_BC5_UNORM;
    case Format::BC5_SNORM:
        return DXGI_FORMAT_BC5_SNORM;
    case Format::BC6H_UF16:
        return DXGI_FORMAT_BC6H_UF16;
    case Format::BC6H_SF16:
        return DXGI_FORMAT_BC6H_SF16;
    case Format::BC7_UNORM:
        return DXGI_FORMAT_BC7_UNORM;
    case Format::BC7_UNORM_SRGB:
        return DXGI_FORMAT_BC7_UNORM_SRGB;

    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

/* static */ DXGI_FORMAT
D3DUtil::calcResourceFormat(UsageType usage, Int usageFlags, DXGI_FORMAT format)
{
    SLANG_UNUSED(usage);
    if (usageFlags)
    {
        switch (format)
        {
        case DXGI_FORMAT_R32_FLOAT: /* fallthru */
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_D32_FLOAT:
            {
                return DXGI_FORMAT_R32_TYPELESS;
            }
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
            return DXGI_FORMAT_R24G8_TYPELESS;
        default:
            break;
        }
        return format;
    }
    return format;
}

/* static */ DXGI_FORMAT D3DUtil::calcFormat(UsageType usage, DXGI_FORMAT format)
{
    switch (usage)
    {
    case USAGE_COUNT_OF:
    case USAGE_UNKNOWN:
        {
            return DXGI_FORMAT_UNKNOWN;
        }
    case USAGE_DEPTH_STENCIL:
        {
            switch (format)
            {
            case DXGI_FORMAT_D32_FLOAT: /* fallthru */
            case DXGI_FORMAT_R32_TYPELESS:
                {
                    return DXGI_FORMAT_D32_FLOAT;
                }
            case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
                return DXGI_FORMAT_D24_UNORM_S8_UINT;
            case DXGI_FORMAT_R24G8_TYPELESS:
                return DXGI_FORMAT_D24_UNORM_S8_UINT;
            default:
                break;
            }
            return format;
        }
    case USAGE_TARGET:
        {
            switch (format)
            {
            case DXGI_FORMAT_D32_FLOAT: /* fallthru */
            case DXGI_FORMAT_D24_UNORM_S8_UINT:
                {
                    return DXGI_FORMAT_UNKNOWN;
                }
            case DXGI_FORMAT_R32_TYPELESS:
                return DXGI_FORMAT_R32_FLOAT;
            default:
                break;
            }
            return format;
        }
    case USAGE_SRV:
        {
            switch (format)
            {
            case DXGI_FORMAT_D32_FLOAT: /* fallthru */
            case DXGI_FORMAT_R32_TYPELESS:
                {
                    return DXGI_FORMAT_R32_FLOAT;
                }
            case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
                return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            default:
                break;
            }

            return format;
        }
    }

    assert(!"Not reachable");
    return DXGI_FORMAT_UNKNOWN;
}

bool D3DUtil::isTypeless(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC7_TYPELESS:
        {
            return true;
        }
    default:
        break;
    }
    return false;
}

/* static */ Int D3DUtil::getNumColorChannelBits(DXGI_FORMAT fmt)
{
    switch (fmt)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
        {
            return 32;
        }
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
        {
            return 16;
        }
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
        {
            return 10;
        }
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        {
            return 8;
        }
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
        {
            return 5;
        }
    case DXGI_FORMAT_B4G4R4A4_UNORM:
        return 4;

    default:
        return 0;
    }
}

// Note: this subroutine is now only used by D3D11 for generating bytecode to go into input layouts.
//
// TODO: we can probably remove that code completely by switching to a PSO-like model across all
// APIs.
//
/* static */ Result D3DUtil::compileHLSLShader(
    char const* sourcePath,
    char const* source,
    char const* entryPointName,
    char const* dxProfileName,
    ComPtr<ID3DBlob>& shaderBlobOut)
{
#if !SLANG_ENABLE_DXBC_SUPPORT
    return SLANG_E_NOT_IMPLEMENTED;
#else
    // Rather than statically link against the `d3dcompile` library, we
    // dynamically load it.
    //
    // Note: A more realistic application would compile from HLSL text to D3D
    // shader bytecode as part of an offline process, rather than doing it
    // on-the-fly like this
    //
    static pD3DCompile compileFunc = nullptr;
    if (!compileFunc)
    {
        // On Linux, vkd3d-utils isn't suitable as a unix replacement for fxc
        // due to at least the following missing feature:
        // https://bugs.winehq.org/show_bug.cgi?id=54872

        // TODO(tfoley): maybe want to search for one of a few versions of the DLL
        const char* const libName = "d3dcompiler_47";
        SharedLibrary::Handle compilerModule;
        if (SLANG_FAILED(SharedLibrary::load(libName, compilerModule)))
        {
            fprintf(stderr, "error: failed to load '%s'\n", libName);
            return SLANG_FAIL;
        }

        compileFunc =
            (pD3DCompile)SharedLibrary::findSymbolAddressByName(compilerModule, "D3DCompile");
        if (!compileFunc)
        {
            fprintf(stderr, "error: failed load symbol 'D3DCompile'\n");
            return SLANG_FAIL;
        }
    }

    // For this example, we turn on debug output, and turn off all
    // optimization. A real application would only use these flags
    // when shader debugging is needed.
    UINT flags = 0;
    flags |= D3DCOMPILE_DEBUG;
    flags |= D3DCOMPILE_OPTIMIZATION_LEVEL0 | D3DCOMPILE_SKIP_OPTIMIZATION;

    // We will always define `__HLSL__` when compiling here, so that
    // input code can react differently to being compiled as pure HLSL.
    D3D_SHADER_MACRO defines[] = {
        {"__HLSL__", "1"},
        {nullptr, nullptr},
    };

    // The `D3DCompile` entry point takes a bunch of parameters, but we
    // don't really need most of them for Slang-generated code.
    ComPtr<ID3DBlob> shaderBlob;
    ComPtr<ID3DBlob> errorBlob;

    HRESULT hr = compileFunc(
        source,
        strlen(source),
        sourcePath,
        &defines[0],
        nullptr,
        entryPointName,
        dxProfileName,
        flags,
        0,
        shaderBlob.writeRef(),
        errorBlob.writeRef());

    // If the HLSL-to-bytecode compilation produced any diagnostic messages
    // then we will print them out (whether or not the compilation failed).
    if (errorBlob)
    {
        ::fputs((const char*)errorBlob->GetBufferPointer(), stderr);
        ::fflush(stderr);
#if SLANG_WINDOWS_FAMILY
        ::OutputDebugStringA((const char*)errorBlob->GetBufferPointer());
#endif
    }

    SLANG_RETURN_ON_FAIL(hr);
    shaderBlobOut.swap(shaderBlob);
    return SLANG_OK;
#endif // SLANG_ENABLE_DXBC_SUPPORT
}

/* static */ SharedLibrary::Handle D3DUtil::getDxgiModule()
{
    const char* const libName = SLANG_ENABLE_DXVK ? "dxvk_dxgi" : "dxgi";

    static SharedLibrary::Handle s_dxgiModule = [&]()
    {
        SharedLibrary::Handle h = nullptr;
        SharedLibrary::load(libName, h);
        if (!h)
        {
            fprintf(stderr, "error: failed to load dll '%s'\n", libName);
        }
        return h;
    }();
    return s_dxgiModule;
}

/* static */ SlangResult D3DUtil::createFactory(
    DeviceCheckFlags flags,
    ComPtr<IDXGIFactory>& outFactory)
{
    auto dxgiModule = getDxgiModule();
    if (!dxgiModule)
    {
        return SLANG_FAIL;
    }

    typedef HRESULT(WINAPI * PFN_DXGI_CREATE_FACTORY)(REFIID riid, void** ppFactory);
    typedef HRESULT(WINAPI * PFN_DXGI_CREATE_FACTORY_2)(UINT Flags, REFIID riid, void** ppFactory);

    {
        auto createFactory2 = (PFN_DXGI_CREATE_FACTORY_2)SharedLibrary::findSymbolAddressByName(
            dxgiModule,
            "CreateDXGIFactory2");
        if (createFactory2)
        {
            UINT dxgiFlags = 0;

            if (flags & DeviceCheckFlag::UseDebug)
            {
                dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
            }

            ComPtr<IDXGIFactory4> factory;
            SLANG_RETURN_ON_FAIL(createFactory2(dxgiFlags, IID_PPV_ARGS(factory.writeRef())));

            outFactory = factory;
            return SLANG_OK;
        }
    }

    {
        auto createFactory = (PFN_DXGI_CREATE_FACTORY)SharedLibrary::findSymbolAddressByName(
            dxgiModule,
            "CreateDXGIFactory");
        if (!createFactory)
        {
            fprintf(stderr, "error: failed load symbol '%s'\n", "CreateDXGIFactory");
            return SLANG_FAIL;
        }
        return createFactory(IID_PPV_ARGS(outFactory.writeRef()));
    }
}

/* static */ SlangResult D3DUtil::findAdapters(
    DeviceCheckFlags flags,
    const AdapterLUID* adapterLUID,
    List<ComPtr<IDXGIAdapter>>& outDxgiAdapters)
{
    ComPtr<IDXGIFactory> factory;
    SLANG_RETURN_ON_FAIL(createFactory(flags, factory));
    return findAdapters(flags, adapterLUID, factory, outDxgiAdapters);
}

/* static */ AdapterLUID D3DUtil::getAdapterLUID(IDXGIAdapter* dxgiAdapter)
{
    DXGI_ADAPTER_DESC desc;
    dxgiAdapter->GetDesc(&desc);
    AdapterLUID luid = {};
    SLANG_ASSERT(sizeof(AdapterLUID) >= sizeof(LUID));
    memcpy(&luid, &desc.AdapterLuid, sizeof(LUID));
    return luid;
}

/* static */ bool D3DUtil::isWarp(IDXGIFactory* dxgiFactory, IDXGIAdapter* adapterIn)
{
    ComPtr<IDXGIFactory4> dxgiFactory4;
    if (SLANG_SUCCEEDED(dxgiFactory->QueryInterface(IID_PPV_ARGS(dxgiFactory4.writeRef()))))
    {
        ComPtr<IDXGIAdapter> warpAdapter;
        dxgiFactory4->EnumWarpAdapter(IID_PPV_ARGS(warpAdapter.writeRef()));

        return adapterIn == warpAdapter;
    }

    return false;
}

bool D3DUtil::isUAVBinding(slang::BindingType bindingType)
{
    switch (bindingType)
    {
    case slang::BindingType::MutableRawBuffer:
    case slang::BindingType::MutableTexture:
    case slang::BindingType::MutableTypedBuffer:
        return true;
    default:
        return false;
    }
}

int D3DUtil::getShaderModelFromProfileName(const char* name)
{
    UnownedStringSlice nameSlice(name);

    if (nameSlice.endsWith("5_1"))
        return D3D_SHADER_MODEL_5_1;
    if (nameSlice.endsWith("6_0"))
        return D3D_SHADER_MODEL_6_0;
    if (nameSlice.endsWith("6_1"))
        return D3D_SHADER_MODEL_6_1;
    if (nameSlice.endsWith("6_2"))
        return D3D_SHADER_MODEL_6_2;
    if (nameSlice.endsWith("6_3"))
        return D3D_SHADER_MODEL_6_3;
    if (nameSlice.endsWith("6_4"))
        return D3D_SHADER_MODEL_6_4;
    if (nameSlice.endsWith("6_5"))
        return D3D_SHADER_MODEL_6_5;
    if (nameSlice.endsWith("6_6"))
        return D3D_SHADER_MODEL_6_6;
    if (nameSlice.endsWith("6_7"))
        return 0x67;
    return 0;
}

uint32_t D3DUtil::getPlaneSliceCount(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return 2;
    default:
        return 1;
    }
}

uint32_t D3DUtil::getPlaneSlice(DXGI_FORMAT format, TextureAspect aspect)
{
    switch (aspect)
    {
    case TextureAspect::Default:
    case TextureAspect::Color:
        return 0;
    case TextureAspect::Depth:
        return 0;
    case TextureAspect::Stencil:
        switch (format)
        {
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            return 1;
        default:
            return 0;
        }
    case TextureAspect::Plane0:
        return 0;
    case TextureAspect::Plane1:
        return 1;
    case TextureAspect::Plane2:
        return 2;
    default:
        SLANG_ASSERT_FAILURE("Unknown texture aspect.");
        return 0;
    }
}

D3D12_INPUT_CLASSIFICATION D3DUtil::getInputSlotClass(InputSlotClass slotClass)
{
    switch (slotClass)
    {
    case InputSlotClass::PerVertex:
        return D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    case InputSlotClass::PerInstance:
        return D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
    default:
        SLANG_ASSERT_FAILURE("Unknown input slot class.");
        return D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    }
}

D3D12_FILL_MODE D3DUtil::getFillMode(FillMode mode)
{
    switch (mode)
    {
    case FillMode::Solid:
        return D3D12_FILL_MODE_SOLID;
    case FillMode::Wireframe:
        return D3D12_FILL_MODE_WIREFRAME;
    default:
        SLANG_ASSERT_FAILURE("Unknown fill mode.");
        return D3D12_FILL_MODE_SOLID;
    }
}

D3D12_CULL_MODE D3DUtil::getCullMode(CullMode mode)
{
    switch (mode)
    {
    case CullMode::None:
        return D3D12_CULL_MODE_NONE;
    case CullMode::Front:
        return D3D12_CULL_MODE_FRONT;
    case CullMode::Back:
        return D3D12_CULL_MODE_BACK;
    default:
        SLANG_ASSERT_FAILURE("Unknown cull mode.");
        return D3D12_CULL_MODE_NONE;
    }
}

D3D12_BLEND_OP D3DUtil::getBlendOp(BlendOp op)
{
    switch (op)
    {
    case BlendOp::Add:
        return D3D12_BLEND_OP_ADD;
    case BlendOp::Subtract:
        return D3D12_BLEND_OP_SUBTRACT;
    case BlendOp::ReverseSubtract:
        return D3D12_BLEND_OP_REV_SUBTRACT;
    case BlendOp::Min:
        return D3D12_BLEND_OP_MIN;
    case BlendOp::Max:
        return D3D12_BLEND_OP_MAX;
    default:
        SLANG_ASSERT_FAILURE("Unknown blend op.");
        return D3D12_BLEND_OP_ADD;
    }
}

D3D12_BLEND D3DUtil::getBlendFactor(BlendFactor factor)
{
    switch (factor)
    {
    case BlendFactor::Zero:
        return D3D12_BLEND_ZERO;
    case BlendFactor::One:
        return D3D12_BLEND_ONE;
    case BlendFactor::SrcColor:
        return D3D12_BLEND_SRC_COLOR;
    case BlendFactor::InvSrcColor:
        return D3D12_BLEND_INV_SRC_COLOR;
    case BlendFactor::SrcAlpha:
        return D3D12_BLEND_SRC_ALPHA;
    case BlendFactor::InvSrcAlpha:
        return D3D12_BLEND_INV_SRC_ALPHA;
    case BlendFactor::DestAlpha:
        return D3D12_BLEND_DEST_ALPHA;
    case BlendFactor::InvDestAlpha:
        return D3D12_BLEND_INV_DEST_ALPHA;
    case BlendFactor::DestColor:
        return D3D12_BLEND_DEST_COLOR;
    case BlendFactor::InvDestColor:
        return D3D12_BLEND_INV_DEST_COLOR;
    case BlendFactor::SrcAlphaSaturate:
        return D3D12_BLEND_SRC_ALPHA_SAT;
    case BlendFactor::BlendColor:
        return D3D12_BLEND_BLEND_FACTOR;
    case BlendFactor::InvBlendColor:
        return D3D12_BLEND_INV_BLEND_FACTOR;
    case BlendFactor::SecondarySrcColor:
        return D3D12_BLEND_SRC1_COLOR;
    case BlendFactor::InvSecondarySrcColor:
        return D3D12_BLEND_INV_SRC1_COLOR;
    case BlendFactor::SecondarySrcAlpha:
        return D3D12_BLEND_SRC1_ALPHA;
    case BlendFactor::InvSecondarySrcAlpha:
        return D3D12_BLEND_INV_SRC1_ALPHA;
    default:
        SLANG_ASSERT_FAILURE("Unknown blend factor.");
        return D3D12_BLEND_ZERO;
    }
}

uint32_t D3DUtil::getSubresourceIndex(
    uint32_t mipIndex,
    uint32_t arrayIndex,
    uint32_t planeIndex,
    uint32_t mipLevelCount,
    uint32_t arraySize)
{
    return mipIndex + arrayIndex * mipLevelCount + planeIndex * mipLevelCount * arraySize;
}

uint32_t D3DUtil::getSubresourceMipLevel(uint32_t subresourceIndex, uint32_t mipLevelCount)
{
    return subresourceIndex % mipLevelCount;
}

D3D12_RESOURCE_STATES D3DUtil::getResourceState(ResourceState state)
{
    switch (state)
    {
    case ResourceState::Undefined:
        return D3D12_RESOURCE_STATE_COMMON;
    case ResourceState::General:
        return D3D12_RESOURCE_STATE_COMMON;
    case ResourceState::PreInitialized:
        return D3D12_RESOURCE_STATE_COMMON;
    case ResourceState::VertexBuffer:
        return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    case ResourceState::IndexBuffer:
        return D3D12_RESOURCE_STATE_INDEX_BUFFER;
    case ResourceState::ConstantBuffer:
        return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    case ResourceState::StreamOutput:
        return D3D12_RESOURCE_STATE_STREAM_OUT;
    case ResourceState::ShaderResource:
    case ResourceState::AccelerationStructureBuildInput:
        return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case ResourceState::PixelShaderResource:
        return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    case ResourceState::NonPixelShaderResource:
        return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case ResourceState::UnorderedAccess:
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case ResourceState::RenderTarget:
        return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case ResourceState::DepthRead:
        return D3D12_RESOURCE_STATE_DEPTH_READ;
    case ResourceState::DepthWrite:;
        return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case ResourceState::Present:
        return D3D12_RESOURCE_STATE_PRESENT;
    case ResourceState::IndirectArgument:
        return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    case ResourceState::CopySource:
        return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case ResourceState::CopyDestination:
        return D3D12_RESOURCE_STATE_COPY_DEST;
    case ResourceState::ResolveSource:
        return D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    case ResourceState::ResolveDestination:
        return D3D12_RESOURCE_STATE_RESOLVE_DEST;
    case ResourceState::AccelerationStructure:
        return D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    default:
        return D3D12_RESOURCE_STATE_COMMON;
    }
}

/* static */ SlangResult D3DUtil::reportLiveObjects()
{
    static IDXGIDebug* dxgiDebug = nullptr;

#if SLANG_ENABLE_DXGI_DEBUG
    if (!dxgiDebug)
    {
        HMODULE debugModule = LoadLibraryA("dxgidebug.dll");
        if (debugModule != INVALID_HANDLE_VALUE)
        {
            auto fun = reinterpret_cast<decltype(&DXGIGetDebugInterface)>(
                GetProcAddress(debugModule, "DXGIGetDebugInterface"));
            if (fun)
            {
                fun(__uuidof(IDXGIDebug), (void**)&dxgiDebug);
            }
        }
    }
#endif

    if (dxgiDebug)
    {
        const GUID DXGI_DEBUG_ALL_ =
            {0xe48ae283, 0xda80, 0x490b, {0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x8}};
        dxgiDebug->ReportLiveObjects(DXGI_DEBUG_ALL_, DXGI_DEBUG_RLO_ALL);
        return SLANG_OK;
    }

    return SLANG_E_NOT_AVAILABLE;
}

Result SLANG_MCALL reportD3DLiveObjects()
{
    return D3DUtil::reportLiveObjects();
}


/* static */ SlangResult D3DUtil::waitForCrashDumpCompletion(HRESULT res)
{
    // If it's not a device remove/reset then theres nothing to wait for
    if (!(res == DXGI_ERROR_DEVICE_REMOVED || res == DXGI_ERROR_DEVICE_RESET))
    {
        return SLANG_OK;
    }

#if GFX_NV_AFTERMATH
    {
        GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
        if (GFSDK_Aftermath_GetCrashDumpStatus(&status) != GFSDK_Aftermath_Result_Success)
        {
            return SLANG_FAIL;
        }

        const auto startTick = Process::getClockTick();
        const auto frequency = Process::getClockFrequency();

        float timeOutInSecs = 1.0f;

        uint64_t timeOutTicks = uint64_t(frequency * timeOutInSecs) + 1;

        // Loop while Aftermath crash dump data collection has not finished or
        // the application is still processing the crash dump data.
        while (status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed &&
               status != GFSDK_Aftermath_CrashDump_Status_Finished &&
               Process::getClockTick() - startTick < timeOutTicks)
        {
            // Sleep a couple of milliseconds and poll the status again.
            Process::sleepCurrentThread(50);
            if (GFSDK_Aftermath_GetCrashDumpStatus(&status) != GFSDK_Aftermath_Result_Success)
            {
                return SLANG_FAIL;
            }
        }

        if (status == GFSDK_Aftermath_CrashDump_Status_Finished)
        {
            return SLANG_OK;
        }
        else
        {
            return SLANG_E_TIME_OUT;
        }
    }
#endif

    return SLANG_OK;
}

/* static */ SlangResult D3DUtil::findAdapters(
    DeviceCheckFlags flags,
    const AdapterLUID* adapterLUID,
    IDXGIFactory* dxgiFactory,
    List<ComPtr<IDXGIAdapter>>& outDxgiAdapters)
{
    outDxgiAdapters.clear();

    ComPtr<IDXGIAdapter> warpAdapter;
    if ((flags & DeviceCheckFlag::UseHardwareDevice) == 0)
    {
        ComPtr<IDXGIFactory4> dxgiFactory4;
        if (SLANG_SUCCEEDED(dxgiFactory->QueryInterface(IID_PPV_ARGS(dxgiFactory4.writeRef()))))
        {
            dxgiFactory4->EnumWarpAdapter(IID_PPV_ARGS(warpAdapter.writeRef()));
            if (!adapterLUID || D3DUtil::getAdapterLUID(warpAdapter) == *adapterLUID)
            {
                outDxgiAdapters.add(warpAdapter);
            }
        }
    }

    for (UINT adapterIndex = 0; true; adapterIndex++)
    {
        ComPtr<IDXGIAdapter> dxgiAdapter;
        if (dxgiFactory->EnumAdapters(adapterIndex, dxgiAdapter.writeRef()) == DXGI_ERROR_NOT_FOUND)
            break;

        // Skip if warp (as we will have already added it)
        if (dxgiAdapter == warpAdapter)
        {
            continue;
        }
        if (adapterLUID && D3DUtil::getAdapterLUID(dxgiAdapter) != *adapterLUID)
        {
            continue;
        }

        // Get if it's software
        UINT deviceFlags = 0;
        ComPtr<IDXGIAdapter1> dxgiAdapter1;
        if (SLANG_SUCCEEDED(dxgiAdapter->QueryInterface(IID_PPV_ARGS(dxgiAdapter1.writeRef()))))
        {
            DXGI_ADAPTER_DESC1 desc;
            dxgiAdapter1->GetDesc1(&desc);
            deviceFlags = desc.Flags;
        }

        // If the right type then add it
        if ((deviceFlags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
            (flags & DeviceCheckFlag::UseHardwareDevice) != 0)
        {
            outDxgiAdapters.add(dxgiAdapter);
        }
    }

    return SLANG_OK;
}

#if SLANG_GFX_HAS_DXR_SUPPORT
Result D3DAccelerationStructureInputsBuilder::build(
    const IAccelerationStructure::BuildInputs& buildInputs,
    IDebugCallback* callback)
{
    if (buildInputs.geometryDescs)
    {
        geomDescs.setCount(buildInputs.descCount);
        for (Index i = 0; i < geomDescs.getCount(); i++)
        {
            auto& inputGeomDesc = buildInputs.geometryDescs[i];
            geomDescs[i].Flags = translateGeometryFlags(inputGeomDesc.flags);
            switch (inputGeomDesc.type)
            {
            case IAccelerationStructure::GeometryType::Triangles:
                geomDescs[i].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                geomDescs[i].Triangles.IndexBuffer = inputGeomDesc.content.triangles.indexData;
                geomDescs[i].Triangles.IndexCount = inputGeomDesc.content.triangles.indexCount;
                geomDescs[i].Triangles.IndexFormat =
                    D3DUtil::getMapFormat(inputGeomDesc.content.triangles.indexFormat);
                geomDescs[i].Triangles.Transform3x4 = inputGeomDesc.content.triangles.transform3x4;
                geomDescs[i].Triangles.VertexBuffer.StartAddress =
                    inputGeomDesc.content.triangles.vertexData;
                geomDescs[i].Triangles.VertexBuffer.StrideInBytes =
                    inputGeomDesc.content.triangles.vertexStride;
                geomDescs[i].Triangles.VertexCount = inputGeomDesc.content.triangles.vertexCount;
                geomDescs[i].Triangles.VertexFormat =
                    D3DUtil::getMapFormat(inputGeomDesc.content.triangles.vertexFormat);
                break;
            case IAccelerationStructure::GeometryType::ProcedurePrimitives:
                geomDescs[i].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
                geomDescs[i].AABBs.AABBCount = inputGeomDesc.content.proceduralAABBs.count;
                geomDescs[i].AABBs.AABBs.StartAddress = inputGeomDesc.content.proceduralAABBs.data;
                geomDescs[i].AABBs.AABBs.StrideInBytes =
                    inputGeomDesc.content.proceduralAABBs.stride;
                break;
            default:
                callback->handleMessage(
                    DebugMessageType::Error,
                    DebugMessageSource::Layer,
                    "invalid value of IAccelerationStructure::GeometryType.");
                return SLANG_E_INVALID_ARG;
            }
        }
    }
    desc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    desc.NumDescs = buildInputs.descCount;
    switch (buildInputs.kind)
    {
    case IAccelerationStructure::Kind::TopLevel:
        desc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        desc.InstanceDescs = buildInputs.instanceDescs;
        break;
    case IAccelerationStructure::Kind::BottomLevel:
        desc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        desc.pGeometryDescs = geomDescs.getBuffer();
        break;
    }
    desc.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)buildInputs.flags;
    return SLANG_OK;
}
#endif

} // namespace gfx
