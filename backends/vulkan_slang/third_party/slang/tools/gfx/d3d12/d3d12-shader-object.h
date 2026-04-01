// d3d12-shader-object.h
#pragma once

#include "d3d12-base.h"
#include "d3d12-helper-functions.h"
#include "d3d12-submitter.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

struct DescriptorTable
{
    DescriptorHeapReference m_heap;
    uint32_t m_offset = 0;
    uint32_t m_count = 0;

    SLANG_FORCE_INLINE uint32_t getDescriptorCount() const { return m_count; }

    /// Get the GPU handle at the specified index
    SLANG_FORCE_INLINE D3D12_GPU_DESCRIPTOR_HANDLE getGpuHandle(uint32_t index = 0) const
    {
        SLANG_ASSERT(index < getDescriptorCount());
        return m_heap.getGpuHandle(m_offset + index);
    }

    /// Get the CPU handle at the specified index
    SLANG_FORCE_INLINE D3D12_CPU_DESCRIPTOR_HANDLE getCpuHandle(uint32_t index = 0) const
    {
        SLANG_ASSERT(index < getDescriptorCount());
        return m_heap.getCpuHandle(m_offset + index);
    }

    void freeIfSupported()
    {
        if (m_count)
        {
            m_heap.freeIfSupported(m_offset, m_count);
            m_offset = 0;
            m_count = 0;
        }
    }

    bool allocate(uint32_t count)
    {
        auto allocatedOffset = m_heap.allocate(count);
        if (allocatedOffset == -1)
            return false;
        m_offset = allocatedOffset;
        m_count = count;
        return true;
    }

    bool allocate(DescriptorHeapReference heap, uint32_t count)
    {
        auto allocatedOffset = heap.allocate(count);
        if (allocatedOffset == -1)
            return false;
        m_heap = heap;
        m_offset = allocatedOffset;
        m_count = count;
        return true;
    }
};

/// A reprsentation of an allocated descriptor set, consisting of an option resource table and
/// an optional sampler table
struct DescriptorSet
{
    DescriptorTable resourceTable;
    DescriptorTable samplerTable;

    void freeIfSupported()
    {
        resourceTable.freeIfSupported();
        samplerTable.freeIfSupported();
    }
};

class ShaderObjectImpl
    : public ShaderObjectBaseImpl<ShaderObjectImpl, ShaderObjectLayoutImpl, SimpleShaderObjectData>
{
    typedef ShaderObjectBaseImpl<ShaderObjectImpl, ShaderObjectLayoutImpl, SimpleShaderObjectData>
        Super;

public:
    static Result create(
        DeviceImpl* device,
        ShaderObjectLayoutImpl* layout,
        ShaderObjectImpl** outShaderObject);

    ~ShaderObjectImpl();

    RendererBase* getDevice() { return m_device.get(); }

    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint) override;

    virtual SLANG_NO_THROW const void* SLANG_MCALL getRawData() override;

    virtual SLANG_NO_THROW Size SLANG_MCALL getSize() override;

    // TODO: What to do with size_t?
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setData(ShaderOffset const& inOffset, void const* data, size_t inSize) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setObject(ShaderOffset const& offset, IShaderObject* object) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setResource(ShaderOffset const& offset, IResourceView* resourceView) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setSampler(ShaderOffset const& offset, ISamplerState* sampler) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL setCombinedTextureSampler(
        ShaderOffset const& offset,
        IResourceView* textureView,
        ISamplerState* sampler) override;

protected:
    Result init(
        DeviceImpl* device,
        ShaderObjectLayoutImpl* layout,
        DescriptorHeapReference viewHeap,
        DescriptorHeapReference samplerHeap);

    /// Write the uniform/ordinary data of this object into the given `dest` buffer at the given
    /// `offset`
    Result _writeOrdinaryData(
        PipelineCommandEncoder* encoder,
        BufferResourceImpl* buffer,
        Offset offset,
        Size destSize,
        ShaderObjectLayoutImpl* specializedLayout);

    bool shouldAllocateConstantBuffer(TransientResourceHeapImpl* transientHeap);

    /// Ensure that the `m_ordinaryDataBuffer` has been created, if it is needed
    Result _ensureOrdinaryDataBufferCreatedIfNeeded(
        PipelineCommandEncoder* encoder,
        ShaderObjectLayoutImpl* specializedLayout);

public:
    void updateSubObjectsRecursive();
    /// Prepare to bind this object as a parameter block.
    ///
    /// This involves allocating and binding any descriptor tables necessary
    /// to to store the state of the object. The function returns a descriptor
    /// set formed from any table(s) allocated. In addition, the `ioOffset`
    /// parameter will be adjusted to be correct for binding values into
    /// the resulting descriptor set.
    ///
    /// Returns:
    ///   SLANG_OK when successful,
    ///   SLANG_E_OUT_OF_MEMORY when descriptor heap is full.
    ///
    Result prepareToBindAsParameterBlock(
        BindingContext* context,
        BindingOffset& ioOffset,
        ShaderObjectLayoutImpl* specializedLayout,
        DescriptorSet& outDescriptorSet);

    bool checkIfCachedDescriptorSetIsValidRecursive(BindingContext* context);

    /// Bind this object as a `ParameterBlock<X>`
    Result bindAsParameterBlock(
        BindingContext* context,
        BindingOffset const& offset,
        ShaderObjectLayoutImpl* specializedLayout);

    /// Bind this object as a `ConstantBuffer<X>`
    Result bindAsConstantBuffer(
        BindingContext* context,
        DescriptorSet const& descriptorSet,
        BindingOffset const& offset,
        ShaderObjectLayoutImpl* specializedLayout);

    /// Bind this object as a value (for an interface-type parameter)
    Result bindAsValue(
        BindingContext* context,
        DescriptorSet const& descriptorSet,
        BindingOffset const& offset,
        ShaderObjectLayoutImpl* specializedLayout);

    /// Shared logic for `bindAsConstantBuffer()` and `bindAsValue()`
    Result _bindImpl(
        BindingContext* context,
        DescriptorSet const& descriptorSet,
        BindingOffset const& offset,
        ShaderObjectLayoutImpl* specializedLayout);

    Result bindRootArguments(BindingContext* context, uint32_t& index);
    /// A CPU-memory descriptor set holding any descriptors used to represent the
    /// resources/samplers in this object's state
    DescriptorSet m_descriptorSet;
    /// A cached descriptor set on GPU heap.
    DescriptorSet m_cachedGPUDescriptorSet;

    ShortList<RefPtr<Resource>, 8> m_boundResources;
    ShortList<RefPtr<Resource>, 8> m_boundCounterResources;
    List<D3D12_GPU_VIRTUAL_ADDRESS> m_rootArguments;
    /// A constant buffer used to stored ordinary data for this object
    /// and existential-type sub-objects.
    ///
    /// Allocated from transient heap on demand with `_createOrdinaryDataBufferIfNeeded()`
    IBufferResource* m_constantBufferWeakPtr = nullptr;
    Offset m_constantBufferOffset = 0;
    Size m_constantBufferSize = 0;

    /// Dirty bit tracking whether the constant buffer needs to be updated.
    bool m_isConstantBufferDirty = true;
    /// The transient heap from which the constant buffer and descriptor set is allocated.
    TransientResourceHeapImpl* m_cachedTransientHeap;
    /// The version of the transient heap when the constant buffer and descriptor set is
    /// allocated.
    uint64_t m_cachedTransientHeapVersion;

    /// Whether this shader object is allowed to be mutable.
    bool m_isMutable = false;
    /// The version of a mutable shader object.
    uint32_t m_version = 0;
    /// The version of this mutable shader object when the gpu descriptor table is cached.
    uint32_t m_cachedGPUDescriptorSetVersion = -1;
    /// The versions of bound subobjects.
    List<uint32_t> m_subObjectVersions;

    /// Get the layout of this shader object with specialization arguments considered
    ///
    /// This operation should only be called after the shader object has been
    /// fully filled in and finalized.
    ///
    Result getSpecializedLayout(ShaderObjectLayoutImpl** outLayout);

    /// Create the layout for this shader object with specialization arguments considered
    ///
    /// This operation is virtual so that it can be customized by `RootShaderObject`.
    ///
    virtual Result _createSpecializedLayout(ShaderObjectLayoutImpl** outLayout);

    RefPtr<ShaderObjectLayoutImpl> m_specializedLayout;
};

class RootShaderObjectImpl : public ShaderObjectImpl
{
    typedef ShaderObjectImpl Super;

public:
    // Override default reference counting behavior to disable lifetime management via ComPtr.
    // Root objects are managed by command buffer and does not need to be freed by the user.
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 1; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 1; }

public:
    RootShaderObjectLayoutImpl* getLayout();

    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** outEntryPoint) override;
    virtual Result collectSpecializationArgs(ExtendedShaderObjectTypeList& args) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    copyFrom(IShaderObject* object, ITransientResourceHeap* transientHeap) override;

public:
    Result bindAsRoot(BindingContext* context, RootShaderObjectLayoutImpl* specializedLayout);

public:
    Result init(DeviceImpl* device) { return SLANG_OK; }

    Result resetImpl(
        DeviceImpl* device,
        RootShaderObjectLayoutImpl* layout,
        DescriptorHeapReference viewHeap,
        DescriptorHeapReference samplerHeap,
        bool isMutable);

    Result reset(
        DeviceImpl* device,
        RootShaderObjectLayoutImpl* layout,
        TransientResourceHeapImpl* heap);

protected:
    virtual Result _createSpecializedLayout(ShaderObjectLayoutImpl** outLayout) override;

    List<RefPtr<ShaderObjectImpl>> m_entryPoints;
};

class MutableRootShaderObjectImpl : public RootShaderObjectImpl
{
public:
    // Enable reference counting.
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return ShaderObjectBase::addRef(); }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return ShaderObjectBase::release(); }
};

} // namespace d3d12
} // namespace gfx
