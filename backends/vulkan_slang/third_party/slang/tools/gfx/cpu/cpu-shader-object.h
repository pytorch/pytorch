// cpu-shader-object.h
#pragma once
#include "cpu-base.h"
#include "cpu-shader-object-layout.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

class CPUShaderObjectData
{
public:
    Slang::List<char> m_ordinaryData;
    // Any "ordinary" / uniform data for this object
    Slang::RefPtr<BufferResourceImpl> m_bufferResource;
    Slang::RefPtr<BufferResourceViewImpl> m_bufferView;

    Index getCount();
    void setCount(Index count);
    char* getBuffer();

    ~CPUShaderObjectData();

    /// Returns a StructuredBuffer resource view for GPU access into the buffer content.
    /// Creates a StructuredBuffer resource if it has not been created.
    ResourceViewBase* getResourceView(
        RendererBase* device,
        slang::TypeLayoutReflection* elementLayout,
        slang::BindingType bindingType);
};

class ShaderObjectImpl
    : public ShaderObjectBaseImpl<ShaderObjectImpl, ShaderObjectLayoutImpl, CPUShaderObjectData>
{
    typedef ShaderObjectBaseImpl<ShaderObjectImpl, ShaderObjectLayoutImpl, CPUShaderObjectData>
        Super;

public:
    List<RefPtr<ResourceViewImpl>> m_resources;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    init(IDevice* device, ShaderObjectLayoutImpl* typeLayout);

    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint) override;

    virtual SLANG_NO_THROW const void* SLANG_MCALL getRawData() override;

    virtual SLANG_NO_THROW size_t SLANG_MCALL getSize() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setData(ShaderOffset const& offset, void const* data, size_t size) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setResource(ShaderOffset const& offset, IResourceView* inView) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setObject(ShaderOffset const& offset, IShaderObject* object) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setSampler(ShaderOffset const& offset, ISamplerState* sampler) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL setCombinedTextureSampler(
        ShaderOffset const& offset,
        IResourceView* textureView,
        ISamplerState* sampler) override;

    char* getDataBuffer();
};

class MutableShaderObjectImpl
    : public MutableShaderObject<MutableShaderObjectImpl, ShaderObjectLayoutImpl>
{
};

class EntryPointShaderObjectImpl : public ShaderObjectImpl
{
public:
    EntryPointLayoutImpl* getLayout();
};

class RootShaderObjectImpl : public ShaderObjectImpl
{
public:
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override;

    // An overload for the `init` virtual function, with a more specific type
    Result init(IDevice* device, RootShaderObjectLayoutImpl* programLayout);
    using ShaderObjectImpl::init;

    RootShaderObjectLayoutImpl* getLayout();

    EntryPointShaderObjectImpl* getEntryPoint(Index index);
    List<RefPtr<EntryPointShaderObjectImpl>> m_entryPoints;

    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint) override;
    virtual Result collectSpecializationArgs(ExtendedShaderObjectTypeList& args) override;
};

} // namespace cpu
} // namespace gfx
