// d3d11-helper-functions.h
#pragma once

#include "../../../source/core/slang-list.h"
#include "d3d11-base.h"
#include "slang-gfx.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{
/// Contextual data and operations required when binding shader objects to the pipeline state
struct BindingContext
{
    // One key service that the `BindingContext` provides is abstracting over
    // the difference between the D3D11 compute and graphics/rasteriation pipelines.
    // D3D11 has distinct operations for, e.g., `CSSetShaderResources`
    // for compute vs. `VSSetShaderResources` and `PSSetShaderResources`
    // for rasterization.
    //
    // The context type provides simple operations for setting each class
    // of resource/sampler, which will be overridden in derived types.
    //
    // TODO: These operations should really support binding multiple resources/samplers
    // in one call, so that we can eventually make more efficient use of the API.
    //
    // TODO: We could reasonably also just store the bound resources into
    // lcoal arrays like we are doing for UAVs, and remove the pipeline-specific
    // virtual functions. However, doing so would seemingly eliminate any
    // chance of avoiding redundant binding work when binding changes are
    // made for a root shader object.
    //
    virtual void setCBV(UINT index, ID3D11Buffer* buffer) = 0;
    virtual void setSRV(UINT index, ID3D11ShaderResourceView* srv) = 0;
    virtual void setSampler(UINT index, ID3D11SamplerState* sampler) = 0;

    // Unordered Access Views (UAVs) are a somewhat special case in that
    // the D3D11 API requires them to all be set at once, rather than one
    // at a time. To support this, we will keep a local array of the UAVs
    // that have been bound (up to the maximum supported by D3D 11.0)
    //
    void setUAV(UINT index, ID3D11UnorderedAccessView* uav)
    {
        uavs[index] = uav;

        // We will also track the total number of UAV slots that will
        // need to be bound (including any gaps that might occur due
        // to either explicit bindings or RTV bindings that conflict
        // with the `u` registers for fragment shaders).
        //
        if (uavCount <= index)
        {
            uavCount = index + 1;
        }
    }

    /// The values bound for any UAVs
    ID3D11UnorderedAccessView* uavs[D3D11_PS_CS_UAV_REGISTER_COUNT];

    /// The number of entries in `uavs` that need to be considered when binding to the pipeline
    UINT uavCount = 0;

    /// The D3D11 device that we are using for binding
    DeviceImpl* device = nullptr;

    /// The D3D11 device context that we are using for binding
    ID3D11DeviceContext* context = nullptr;

    /// Initialize a binding context for binding to the given `device` and `context`
    BindingContext(DeviceImpl* device, ID3D11DeviceContext* context)
        : device(device), context(context)
    {
        memset(uavs, 0, sizeof(uavs));
    }
};

/// A `BindingContext` for binding to the compute pipeline
struct ComputeBindingContext : BindingContext
{
    /// Initialize a binding context for binding to the given `device` and `context`
    ComputeBindingContext(DeviceImpl* device, ID3D11DeviceContext* context)
        : BindingContext(device, context)
    {
    }

    void setCBV(UINT index, ID3D11Buffer* buffer) SLANG_OVERRIDE
    {
        context->CSSetConstantBuffers(index, 1, &buffer);
    }

    void setSRV(UINT index, ID3D11ShaderResourceView* srv) SLANG_OVERRIDE
    {
        context->CSSetShaderResources(index, 1, &srv);
    }

    void setSampler(UINT index, ID3D11SamplerState* sampler) SLANG_OVERRIDE
    {
        context->CSSetSamplers(index, 1, &sampler);
    }
};

/// A `BindingContext` for binding to the graphics/rasterization pipeline
struct GraphicsBindingContext : BindingContext
{
    /// Initialize a binding context for binding to the given `device` and `context`
    GraphicsBindingContext(DeviceImpl* device, ID3D11DeviceContext* context)
        : BindingContext(device, context)
    {
    }

    // TODO: The operations here are only dealing with vertex and fragment
    // shaders for now. We should eventually extend them to handle HS/DS/GS
    // bindings. (We might want to skip those stages depending on whether
    // the associated program uses them at all).
    //
    // TODO: If we support cases where different stages might use distinct
    // entry-point parameters, we might need to support some modes where
    // a "stage mask" is passed in that applies to the bindings.
    //
    void setCBV(UINT index, ID3D11Buffer* buffer) SLANG_OVERRIDE
    {
        context->VSSetConstantBuffers(index, 1, &buffer);
        context->PSSetConstantBuffers(index, 1, &buffer);
    }

    void setSRV(UINT index, ID3D11ShaderResourceView* srv) SLANG_OVERRIDE
    {
        context->VSSetShaderResources(index, 1, &srv);
        context->PSSetShaderResources(index, 1, &srv);
    }

    void setSampler(UINT index, ID3D11SamplerState* sampler) SLANG_OVERRIDE
    {
        context->VSSetSamplers(index, 1, &sampler);
        context->PSSetSamplers(index, 1, &sampler);
    }
};

// In order to bind shader parameters to the correct locations, we need to
// be able to describe those locations. Most shader parameters will
// only consume a single type of D3D11-visible regsiter (e.g., a `t`
// register for a txture, or an `s` register for a sampler), and scalar
// integers suffice for these cases.
//
// In more complex cases we might be binding an entire "sub-object" like
// a parameter block, an entry point, etc. For the general case, we need
// to be able to represent a composite offset that includes offsets for
// each of the register classes known to D3D11.

/// A "simple" binding offset that records an offset in CBV/SRV/UAV/Sampler slots
struct SimpleBindingOffset
{
    uint32_t cbv = 0;
    uint32_t srv = 0;
    uint32_t uav = 0;
    uint32_t sampler = 0;

    /// Create a default (zero) offset
    SimpleBindingOffset() {}

    /// Create an offset based on offset information in the given Slang `varLayout`
    SimpleBindingOffset(slang::VariableLayoutReflection* varLayout)
    {
        if (varLayout)
        {
            cbv = (uint32_t)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER);
            srv = (uint32_t)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE);
            uav = (uint32_t)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS);
            sampler = (uint32_t)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_SAMPLER_STATE);
        }
    }

    /// Create an offset based on size/stride information in the given Slang `typeLayout`
    SimpleBindingOffset(slang::TypeLayoutReflection* typeLayout)
    {
        if (typeLayout)
        {
            cbv = (uint32_t)typeLayout->getSize(SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER);
            srv = (uint32_t)typeLayout->getSize(SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE);
            uav = (uint32_t)typeLayout->getSize(SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS);
            sampler = (uint32_t)typeLayout->getSize(SLANG_PARAMETER_CATEGORY_SAMPLER_STATE);
        }
    }

    /// Add any values in the given `offset`
    void operator+=(SimpleBindingOffset const& offset)
    {
        cbv += offset.cbv;
        srv += offset.srv;
        uav += offset.uav;
        sampler += offset.sampler;
    }
};

// While a "simple" binding offset representation will work in many cases,
// once we need to deal with layout for programs with interface-type parameters
// that have been statically specialized, we also need to track the offset
// for where to bind any "pending" data that arises from the process of static
// specialization.
//
// In order to conveniently track both the "primary" and "pending" offset information,
// we will define a more complete `BindingOffset` type that combines simple
// binding offsets for the primary and pending parts.

/// A representation of the offset at which to bind a shader parameter or sub-object
struct BindingOffset : SimpleBindingOffset
{
    // Offsets for "primary" data are stored directly in the `BindingOffset`
    // via the inheritance from `SimpleBindingOffset`.

    /// Offset for any "pending" data
    SimpleBindingOffset pending;

    /// Create a default (zero) offset
    BindingOffset() {}

    /// Create an offset from a simple offset
    explicit BindingOffset(SimpleBindingOffset const& offset)
        : SimpleBindingOffset(offset)
    {
    }

    /// Create an offset based on offset information in the given Slang `varLayout`
    BindingOffset(slang::VariableLayoutReflection* varLayout)
        : SimpleBindingOffset(varLayout), pending(varLayout->getPendingDataLayout())
    {
    }

    /// Create an offset based on size/stride information in the given Slang `typeLayout`
    BindingOffset(slang::TypeLayoutReflection* typeLayout)
        : SimpleBindingOffset(typeLayout), pending(typeLayout->getPendingDataTypeLayout())
    {
    }

    /// Add any values in the given `offset`
    void operator+=(SimpleBindingOffset const& offset) { SimpleBindingOffset::operator+=(offset); }

    /// Add any values in the given `offset`
    void operator+=(BindingOffset const& offset)
    {
        SimpleBindingOffset::operator+=(offset);
        pending += offset.pending;
    }
};

bool isSupportedNVAPIOp(IUnknown* dev, uint32_t op);

D3D11_BIND_FLAG calcResourceFlag(ResourceState state);
int _calcResourceBindFlags(ResourceStateSet allowedStates);
int _calcResourceAccessFlags(MemoryType memType);

D3D11_FILTER_TYPE translateFilterMode(TextureFilteringMode mode);
D3D11_FILTER_REDUCTION_TYPE translateFilterReduction(TextureReductionOp op);
D3D11_TEXTURE_ADDRESS_MODE translateAddressingMode(TextureAddressingMode mode);
D3D11_COMPARISON_FUNC translateComparisonFunc(ComparisonFunc func);

D3D11_STENCIL_OP translateStencilOp(StencilOp op);
D3D11_FILL_MODE translateFillMode(FillMode mode);
D3D11_CULL_MODE translateCullMode(CullMode mode);
bool isBlendDisabled(AspectBlendDesc const& desc);
bool isBlendDisabled(TargetBlendDesc const& desc);
D3D11_BLEND_OP translateBlendOp(BlendOp op);
D3D11_BLEND translateBlendFactor(BlendFactor factor);
D3D11_COLOR_WRITE_ENABLE translateRenderTargetWriteMask(RenderTargetWriteMaskT mask);

void initSrvDesc(
    IResource::Type resourceType,
    const ITextureResource::Desc& textureDesc,
    DXGI_FORMAT pixelFormat,
    D3D11_SHADER_RESOURCE_VIEW_DESC& descOut);
} // namespace d3d11

Result SLANG_MCALL getD3D11Adapters(List<AdapterInfo>& outAdapters);

Result SLANG_MCALL createD3D11Device(const IDevice::Desc* desc, IDevice** outDevice);

} // namespace gfx
