// cuda-shader-object.h
#pragma once
#include "cuda-base.h"
#include "cuda-buffer.h"
#include "cuda-resource-views.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

class ShaderObjectData
{
public:
    bool isHostOnly = false;
    Slang::RefPtr<BufferResourceImpl> m_bufferResource;
    Slang::RefPtr<ResourceViewImpl> m_bufferView;
    Slang::List<uint8_t> m_cpuBuffer;

    Result setCount(Index count);
    Slang::Index getCount();
    void* getBuffer();

    /// Returns a resource view for GPU access into the buffer content.
    ResourceViewBase* getResourceView(
        RendererBase* device,
        slang::TypeLayoutReflection* elementLayout,
        slang::BindingType bindingType);
};

class ShaderObjectImpl
    : public ShaderObjectBaseImpl<ShaderObjectImpl, ShaderObjectLayoutImpl, ShaderObjectData>
{
    typedef ShaderObjectBaseImpl<ShaderObjectImpl, ShaderObjectLayoutImpl, ShaderObjectData> Super;

public:
    List<RefPtr<ResourceViewImpl>> resources;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    init(IDevice* device, ShaderObjectLayoutImpl* typeLayout);

    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint) override;

    virtual SLANG_NO_THROW const void* SLANG_MCALL getRawData() override;

    virtual SLANG_NO_THROW Size SLANG_MCALL getSize() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setData(ShaderOffset const& offset, void const* data, Size size) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setResource(ShaderOffset const& offset, IResourceView* resourceView) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setObject(ShaderOffset const& offset, IShaderObject* object) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setSampler(ShaderOffset const& offset, ISamplerState* sampler) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL setCombinedTextureSampler(
        ShaderOffset const& offset,
        IResourceView* textureView,
        ISamplerState* sampler) override;
};

class MutableShaderObjectImpl
    : public MutableShaderObject<MutableShaderObjectImpl, ShaderObjectLayoutImpl>
{
};

class EntryPointShaderObjectImpl : public ShaderObjectImpl
{
public:
    EntryPointShaderObjectImpl();
};

class RootShaderObjectImpl : public ShaderObjectImpl
{
public:
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override;

public:
    List<RefPtr<EntryPointShaderObjectImpl>> entryPointObjects;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    init(IDevice* device, ShaderObjectLayoutImpl* typeLayout) override;
    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint) override;
    virtual Result collectSpecializationArgs(ExtendedShaderObjectTypeList& args) override;
};

} // namespace cuda
#endif
} // namespace gfx
