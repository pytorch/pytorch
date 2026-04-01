// d3d11-helper-functions.cpp
#include "d3d11-helper-functions.h"

#include "d3d11-device.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{
bool isSupportedNVAPIOp(IUnknown* dev, uint32_t op)
{
#ifdef GFX_NVAPI
    {
        bool isSupported;
        NvAPI_Status status =
            NvAPI_D3D11_IsNvShaderExtnOpCodeSupported(dev, NvU32(op), &isSupported);
        return status == NVAPI_OK && isSupported;
    }
#else
    return false;
#endif
}

D3D11_BIND_FLAG calcResourceFlag(ResourceState state)
{
    switch (state)
    {
    case ResourceState::VertexBuffer:
        return D3D11_BIND_VERTEX_BUFFER;
    case ResourceState::IndexBuffer:
        return D3D11_BIND_INDEX_BUFFER;
    case ResourceState::ConstantBuffer:
        return D3D11_BIND_CONSTANT_BUFFER;
    case ResourceState::StreamOutput:
        return D3D11_BIND_STREAM_OUTPUT;
    case ResourceState::RenderTarget:
        return D3D11_BIND_RENDER_TARGET;
    case ResourceState::DepthRead:
    case ResourceState::DepthWrite:
        return D3D11_BIND_DEPTH_STENCIL;
    case ResourceState::UnorderedAccess:
        return D3D11_BIND_UNORDERED_ACCESS;
    case ResourceState::ShaderResource:
    case ResourceState::PixelShaderResource:
    case ResourceState::NonPixelShaderResource:
        return D3D11_BIND_SHADER_RESOURCE;
    default:
        return D3D11_BIND_FLAG(0);
    }
}

int _calcResourceBindFlags(ResourceStateSet allowedStates)
{
    int dstFlags = 0;
    for (uint32_t i = 0; i < (uint32_t)ResourceState::_Count; i++)
    {
        auto state = (ResourceState)i;
        if (allowedStates.contains(state))
            dstFlags |= calcResourceFlag(state);
    }
    return dstFlags;
}

int _calcResourceAccessFlags(MemoryType memType)
{
    switch (memType)
    {
    case MemoryType::DeviceLocal:
        return 0;
    case MemoryType::ReadBack:
        return D3D11_CPU_ACCESS_READ;
    case MemoryType::Upload:
        return D3D11_CPU_ACCESS_WRITE;
    default:
        assert(!"Invalid flags");
        return 0;
    }
}

D3D11_FILTER_TYPE translateFilterMode(TextureFilteringMode mode)
{
    switch (mode)
    {
    default:
        return D3D11_FILTER_TYPE(0);

#define CASE(SRC, DST)              \
    case TextureFilteringMode::SRC: \
        return D3D11_FILTER_TYPE_##DST

        CASE(Point, POINT);
        CASE(Linear, LINEAR);

#undef CASE
    }
}

D3D11_FILTER_REDUCTION_TYPE translateFilterReduction(TextureReductionOp op)
{
    switch (op)
    {
    default:
        return D3D11_FILTER_REDUCTION_TYPE(0);

#define CASE(SRC, DST)            \
    case TextureReductionOp::SRC: \
        return D3D11_FILTER_REDUCTION_TYPE_##DST

        CASE(Average, STANDARD);
        CASE(Comparison, COMPARISON);
        CASE(Minimum, MINIMUM);
        CASE(Maximum, MAXIMUM);

#undef CASE
    }
}

D3D11_TEXTURE_ADDRESS_MODE translateAddressingMode(TextureAddressingMode mode)
{
    switch (mode)
    {
    default:
        return D3D11_TEXTURE_ADDRESS_MODE(0);

#define CASE(SRC, DST)               \
    case TextureAddressingMode::SRC: \
        return D3D11_TEXTURE_ADDRESS_##DST

        CASE(Wrap, WRAP);
        CASE(ClampToEdge, CLAMP);
        CASE(ClampToBorder, BORDER);
        CASE(MirrorRepeat, MIRROR);
        CASE(MirrorOnce, MIRROR_ONCE);

#undef CASE
    }
}

D3D11_COMPARISON_FUNC translateComparisonFunc(ComparisonFunc func)
{
    switch (func)
    {
    default:
        // TODO: need to report failures
        return D3D11_COMPARISON_ALWAYS;

#define CASE(FROM, TO)         \
    case ComparisonFunc::FROM: \
        return D3D11_COMPARISON_##TO

        CASE(Never, NEVER);
        CASE(Less, LESS);
        CASE(Equal, EQUAL);
        CASE(LessEqual, LESS_EQUAL);
        CASE(Greater, GREATER);
        CASE(NotEqual, NOT_EQUAL);
        CASE(GreaterEqual, GREATER_EQUAL);
        CASE(Always, ALWAYS);
#undef CASE
    }
}

D3D11_STENCIL_OP translateStencilOp(StencilOp op)
{
    switch (op)
    {
    default:
        // TODO: need to report failures
        return D3D11_STENCIL_OP_KEEP;

#define CASE(FROM, TO)    \
    case StencilOp::FROM: \
        return D3D11_STENCIL_OP_##TO

        CASE(Keep, KEEP);
        CASE(Zero, ZERO);
        CASE(Replace, REPLACE);
        CASE(IncrementSaturate, INCR_SAT);
        CASE(DecrementSaturate, DECR_SAT);
        CASE(Invert, INVERT);
        CASE(IncrementWrap, INCR);
        CASE(DecrementWrap, DECR);
#undef CASE
    }
}

D3D11_FILL_MODE translateFillMode(FillMode mode)
{
    switch (mode)
    {
    default:
        // TODO: need to report failures
        return D3D11_FILL_SOLID;

    case FillMode::Solid:
        return D3D11_FILL_SOLID;
    case FillMode::Wireframe:
        return D3D11_FILL_WIREFRAME;
    }
}

D3D11_CULL_MODE translateCullMode(CullMode mode)
{
    switch (mode)
    {
    default:
        // TODO: need to report failures
        return D3D11_CULL_NONE;

    case CullMode::None:
        return D3D11_CULL_NONE;
    case CullMode::Back:
        return D3D11_CULL_BACK;
    case CullMode::Front:
        return D3D11_CULL_FRONT;
    }
}

bool isBlendDisabled(AspectBlendDesc const& desc)
{
    return desc.op == BlendOp::Add && desc.srcFactor == BlendFactor::One &&
           desc.dstFactor == BlendFactor::Zero;
}


bool isBlendDisabled(TargetBlendDesc const& desc)
{
    return isBlendDisabled(desc.color) && isBlendDisabled(desc.alpha);
}

D3D11_BLEND_OP translateBlendOp(BlendOp op)
{
    switch (op)
    {
    default:
        assert(!"unimplemented");
        return (D3D11_BLEND_OP)-1;

#define CASE(FROM, TO)  \
    case BlendOp::FROM: \
        return D3D11_BLEND_OP_##TO
        CASE(Add, ADD);
        CASE(Subtract, SUBTRACT);
        CASE(ReverseSubtract, REV_SUBTRACT);
        CASE(Min, MIN);
        CASE(Max, MAX);
#undef CASE
    }
}

D3D11_BLEND translateBlendFactor(BlendFactor factor)
{
    switch (factor)
    {
    default:
        assert(!"unimplemented");
        return (D3D11_BLEND)-1;

#define CASE(FROM, TO)      \
    case BlendFactor::FROM: \
        return D3D11_BLEND_##TO
        CASE(Zero, ZERO);
        CASE(One, ONE);
        CASE(SrcColor, SRC_COLOR);
        CASE(InvSrcColor, INV_SRC_COLOR);
        CASE(SrcAlpha, SRC_ALPHA);
        CASE(InvSrcAlpha, INV_SRC_ALPHA);
        CASE(DestAlpha, DEST_ALPHA);
        CASE(InvDestAlpha, INV_DEST_ALPHA);
        CASE(DestColor, DEST_COLOR);
        CASE(InvDestColor, INV_DEST_ALPHA);
        CASE(SrcAlphaSaturate, SRC_ALPHA_SAT);
        CASE(BlendColor, BLEND_FACTOR);
        CASE(InvBlendColor, INV_BLEND_FACTOR);
        CASE(SecondarySrcColor, SRC1_COLOR);
        CASE(InvSecondarySrcColor, INV_SRC1_COLOR);
        CASE(SecondarySrcAlpha, SRC1_ALPHA);
        CASE(InvSecondarySrcAlpha, INV_SRC1_ALPHA);
#undef CASE
    }
}

D3D11_COLOR_WRITE_ENABLE translateRenderTargetWriteMask(RenderTargetWriteMaskT mask)
{
    UINT result = 0;
#define CASE(FROM, TO)                              \
    if (mask & RenderTargetWriteMask::Enable##FROM) \
    result |= D3D11_COLOR_WRITE_ENABLE_##TO

    CASE(Red, RED);
    CASE(Green, GREEN);
    CASE(Blue, BLUE);
    CASE(Alpha, ALPHA);

#undef CASE
    return D3D11_COLOR_WRITE_ENABLE(result);
}

void initSrvDesc(
    IResource::Type resourceType,
    const ITextureResource::Desc& textureDesc,
    DXGI_FORMAT pixelFormat,
    D3D11_SHADER_RESOURCE_VIEW_DESC& descOut)
{
    // create SRV
    descOut = D3D11_SHADER_RESOURCE_VIEW_DESC();

    descOut.Format =
        (pixelFormat == DXGI_FORMAT_UNKNOWN)
            ? D3DUtil::calcFormat(D3DUtil::USAGE_SRV, D3DUtil::getMapFormat(textureDesc.format))
            : pixelFormat;
    const int arraySize = calcEffectiveArraySize(textureDesc);
    if (arraySize <= 1)
    {
        switch (textureDesc.type)
        {
        case IResource::Type::Texture1D:
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1D;
            break;
        case IResource::Type::Texture2D:
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            break;
        case IResource::Type::Texture3D:
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
            break;
        default:
            assert(!"Unknown dimension");
        }

        descOut.Texture2D.MipLevels = textureDesc.numMipLevels;
        descOut.Texture2D.MostDetailedMip = 0;
    }
    else if (resourceType == IResource::Type::TextureCube)
    {
        if (textureDesc.arraySize > 1)
        {
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBEARRAY;

            descOut.TextureCubeArray.NumCubes = textureDesc.arraySize;
            descOut.TextureCubeArray.First2DArrayFace = 0;
            descOut.TextureCubeArray.MipLevels = textureDesc.numMipLevels;
            descOut.TextureCubeArray.MostDetailedMip = 0;
        }
        else
        {
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;

            descOut.TextureCube.MipLevels = textureDesc.numMipLevels;
            descOut.TextureCube.MostDetailedMip = 0;
        }
    }
    else
    {
        assert(textureDesc.size.depth > 1 || arraySize > 1);

        switch (textureDesc.type)
        {
        case IResource::Type::Texture1D:
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1DARRAY;
            break;
        case IResource::Type::Texture2D:
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
            break;
        case IResource::Type::Texture3D:
            descOut.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
            break;

        default:
            assert(!"Unknown dimension");
        }

        descOut.Texture2DArray.ArraySize = std::max(textureDesc.size.depth, arraySize);
        descOut.Texture2DArray.MostDetailedMip = 0;
        descOut.Texture2DArray.MipLevels = textureDesc.numMipLevels;
        descOut.Texture2DArray.FirstArraySlice = 0;
    }
}
} // namespace d3d11

Result SLANG_MCALL getD3D11Adapters(List<AdapterInfo>& outAdapters)
{
    List<ComPtr<IDXGIAdapter>> dxgiAdapters;
    SLANG_RETURN_ON_FAIL(
        D3DUtil::findAdapters(DeviceCheckFlag::UseHardwareDevice, nullptr, dxgiAdapters));

    outAdapters.clear();
    for (const auto& dxgiAdapter : dxgiAdapters)
    {
        DXGI_ADAPTER_DESC desc;
        dxgiAdapter->GetDesc(&desc);
        AdapterInfo info = {};
        auto name = String::fromWString(desc.Description);
        memcpy(
            info.name,
            name.getBuffer(),
            Math::Min(name.getLength(), (Index)sizeof(AdapterInfo::name) - 1));
        info.vendorID = desc.VendorId;
        info.deviceID = desc.DeviceId;
        info.luid = D3DUtil::getAdapterLUID(dxgiAdapter);
        outAdapters.add(info);
    }
    return SLANG_OK;
}

Result SLANG_MCALL createD3D11Device(const IDevice::Desc* desc, IDevice** outDevice)
{
    RefPtr<d3d11::DeviceImpl> result = new d3d11::DeviceImpl();
    SLANG_RETURN_ON_FAIL(result->initialize(*desc));
    returnComPtr(outDevice, result);
    return SLANG_OK;
}

} // namespace gfx
