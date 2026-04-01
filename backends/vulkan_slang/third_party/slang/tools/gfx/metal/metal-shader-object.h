// metal-shader-object.h
#pragma once
#include "metal-base.h"
#include "metal-helper-functions.h"
#include "metal-resource-views.h"
#include "metal-sampler.h"
#include "metal-shader-object-layout.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class ShaderObjectImpl
    : public ShaderObjectBaseImpl<ShaderObjectImpl, ShaderObjectLayoutImpl, SimpleShaderObjectData>
{
public:
    static Result create(
        IDevice* device,
        ShaderObjectLayoutImpl* layout,
        ShaderObjectImpl** outShaderObject);

    ~ShaderObjectImpl();

    RendererBase* getDevice() { return m_layout->getDevice(); }

    SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() SLANG_OVERRIDE { return 0; }

    SLANG_NO_THROW Result SLANG_MCALL getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint)
        SLANG_OVERRIDE
    {
        *outEntryPoint = nullptr;
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW const void* SLANG_MCALL getRawData() override
    {
        return m_data.getBuffer();
    }

    virtual SLANG_NO_THROW size_t SLANG_MCALL getSize() override
    {
        return (size_t)m_data.getCount();
    }

    SLANG_NO_THROW Result SLANG_MCALL
    setData(ShaderOffset const& inOffset, void const* data, size_t inSize) SLANG_OVERRIDE;

    SLANG_NO_THROW Result SLANG_MCALL
    setResource(ShaderOffset const& offset, IResourceView* resourceView) SLANG_OVERRIDE;

    SLANG_NO_THROW Result SLANG_MCALL setSampler(ShaderOffset const& offset, ISamplerState* sampler)
        SLANG_OVERRIDE;

    SLANG_NO_THROW Result SLANG_MCALL setCombinedTextureSampler(
        ShaderOffset const& offset,
        IResourceView* textureView,
        ISamplerState* sampler) SLANG_OVERRIDE
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }

public:
protected:
    friend class ProgramVars;

    Result init(IDevice* device, ShaderObjectLayoutImpl* layout);

    /// Write the uniform/ordinary data of this object into the given `dest` buffer at the given
    /// `offset`
    Result _writeOrdinaryData(void* dest, size_t destSize, ShaderObjectLayoutImpl* layout);

    /// Ensure that the `m_ordinaryDataBuffer` has been created, if it is needed
    ///
    /// The `layout` type must represent a specialized layout for this
    /// type that includes any "pending" data.
    ///
    Result _ensureOrdinaryDataBufferCreatedIfNeeded(
        DeviceImpl* device,
        ShaderObjectLayoutImpl* layout);

    BufferResourceImpl* _ensureArgumentBufferUpToDate(
        BindingContext* context,
        DeviceImpl* device,
        ShaderObjectLayoutImpl* layout);

    void writeOrdinaryDataIntoArgumentBuffer(
        slang::TypeLayoutReflection* argumentBufferTypeLayout,
        slang::TypeLayoutReflection* defaultTypeLayout,
        uint8_t* argumentBuffer,
        uint8_t* srcData);

    /// Bind the buffer for ordinary/uniform data, if needed
    ///
    /// The `ioOffset` parameter will be updated to reflect the constant buffer
    /// register consumed by the ordinary data buffer, if one was bound.
    ///
    Result _bindOrdinaryDataBufferIfNeeded(
        BindingContext* context,
        BindingOffset& ioOffset,
        ShaderObjectLayoutImpl* layout);

public:
    /// Bind this object as if it was declared as a `ConstantBuffer<T>` in Slang
    Result bindAsConstantBuffer(
        BindingContext* context,
        BindingOffset const& inOffset,
        ShaderObjectLayoutImpl* layout);

    /// Bind this object as if it was declared as a `ParameterBlock<T>` in Slang
    Result bindAsParameterBlock(
        BindingContext* context,
        BindingOffset const& inOffset,
        ShaderObjectLayoutImpl* layout);

    /// Bind this object as a value that appears in the body of another object.
    ///
    /// This case is directly used when binding an object for an interface-type
    /// sub-object range when static specialization is used. It is also used
    /// indirectly when binding sub-objects to constant buffer or parameter
    /// block ranges.
    ///
    Result bindAsValue(
        BindingContext* context,
        BindingOffset const& offset,
        ShaderObjectLayoutImpl* layout);

    // Because the binding ranges have already been reflected
    // and organized as part of each shader object layout,
    // the object itself can store its data in a small number
    // of simple arrays.

    /// The buffers that are part of the state of this object
    List<RefPtr<BufferResourceViewImpl>> m_buffers;

    /// The textures that are part of the state of this object
    List<RefPtr<TextureResourceViewImpl>> m_textures;

    /// The samplers that are part of the state of this object
    List<RefPtr<SamplerStateImpl>> m_samplers;

    /// A constant buffer used to stored ordinary data for this object
    /// and existential-type sub-objects.
    ///
    /// Created on demand with `_createOrdinaryDataBufferIfNeeded()`
    RefPtr<BufferResourceImpl> m_ordinaryDataBuffer;

    /// Argument buffer created on demand to bind as a parameter block.
    RefPtr<BufferResourceImpl> m_argumentBuffer;


    bool m_isConstantBufferDirty = true;
    bool m_isArgumentBufferDirty = true;
};

class MutableShaderObjectImpl
    : public MutableShaderObject<MutableShaderObjectImpl, ShaderObjectLayoutImpl>
{
};

class RootShaderObjectImpl : public ShaderObjectImpl
{
    typedef ShaderObjectImpl Super;

public:
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 1; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 1; }

    static Result create(
        IDevice* device,
        RootShaderObjectLayoutImpl* layout,
        RootShaderObjectImpl** outShaderObject);

    Result init(IDevice* device, RootShaderObjectLayoutImpl* layout);

    RootShaderObjectLayoutImpl* getLayout()
    {
        return static_cast<RootShaderObjectLayoutImpl*>(m_layout.Ptr());
    }

    GfxCount SLANG_MCALL getEntryPointCount() SLANG_OVERRIDE
    {
        return (GfxCount)m_entryPoints.getCount();
    }
    SlangResult SLANG_MCALL getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint)
        SLANG_OVERRIDE
    {
        returnComPtr(outEntryPoint, m_entryPoints[index]);
        return SLANG_OK;
    }

    virtual Result collectSpecializationArgs(ExtendedShaderObjectTypeList& args) override;

    /// Bind this object as a root shader object
    Result bindAsRoot(BindingContext* context, RootShaderObjectLayoutImpl* specializedLayout);

protected:
    List<RefPtr<ShaderObjectImpl>> m_entryPoints;
};

} // namespace metal
} // namespace gfx
