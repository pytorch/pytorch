#pragma once

#include "core/slang-basic.h"
#include "core/slang-com-object.h"
#include "renderer-shared.h"
#include "slang-gfx.h"

namespace gfx
{
class ShaderObjectLayoutBase;

template<typename T>
class VersionedObjectPool
{
public:
    struct ObjectVersion
    {
        Slang::RefPtr<T> object;
        Slang::RefPtr<TransientResourceHeapBase> transientHeap;
        uint64_t transientHeapVersion;
        bool canRecycle() { return (transientHeap->getVersion() != transientHeapVersion); }
    };
    Slang::List<ObjectVersion> objects;
    SlangInt lastAllocationIndex = -1;
    ObjectVersion& allocate(TransientResourceHeapBase* currentTransientHeap)
    {
        for (SlangInt i = 0; i < objects.getCount(); i++)
        {
            auto& object = objects[i];
            if (object.canRecycle())
            {
                object.transientHeap = currentTransientHeap;
                object.transientHeapVersion = currentTransientHeap->getVersion();
                lastAllocationIndex = i;
                return object;
            }
        }
        ObjectVersion v;
        v.transientHeap = currentTransientHeap;
        v.transientHeapVersion = currentTransientHeap->getVersion();
        objects.add(v);
        lastAllocationIndex = objects.getCount() - 1;
        return objects.getLast();
    }
    ObjectVersion& getLastAllocation() { return objects[lastAllocationIndex]; }
};

class MutableShaderObjectData
{
public:
    // Any "ordinary" / uniform data for this object
    Slang::List<char> m_ordinaryData;

    bool m_dirty = true;

    Slang::Index getCount() { return m_ordinaryData.getCount(); }
    void setCount(Slang::Index count) { m_ordinaryData.setCount(count); }
    char* getBuffer() { return m_ordinaryData.getBuffer(); }
    void markDirty() { m_dirty = true; }

    // We don't actually create any GPU buffers here, since they will be handled
    // by the immutable shader objects once the user calls `getCurrentVersion`.
    ResourceViewBase* getResourceView(
        RendererBase* device,
        slang::TypeLayoutReflection* elementLayout,
        slang::BindingType bindingType)
    {
        return nullptr;
    }
};

template<typename TShaderObject, typename TShaderObjectLayoutImpl>
class MutableShaderObject
    : public ShaderObjectBaseImpl<TShaderObject, TShaderObjectLayoutImpl, MutableShaderObjectData>
{
    typedef ShaderObjectBaseImpl<TShaderObject, TShaderObjectLayoutImpl, MutableShaderObjectData>
        Super;

protected:
    Slang::OrderedDictionary<ShaderOffset, Slang::RefPtr<ResourceViewBase>> m_resources;
    Slang::OrderedDictionary<ShaderOffset, Slang::RefPtr<SamplerStateBase>> m_samplers;
    Slang::OrderedHashSet<ShaderOffset> m_objectOffsets;
    VersionedObjectPool<ShaderObjectBase> m_shaderObjectVersions;
    bool m_dirty = true;
    bool isDirty()
    {
        if (m_dirty)
            return true;
        if (this->m_data.m_dirty)
            return true;
        for (auto& object : this->m_objects)
        {
            if (object && object->isDirty())
                return true;
        }
        return false;
    }

    void markDirty() { m_dirty = true; }

public:
    Result init(RendererBase* device, ShaderObjectLayoutBase* layout)
    {
        this->m_device = device;
        auto layoutImpl = static_cast<TShaderObjectLayoutImpl*>(layout);
        this->m_layout = layoutImpl;
        Slang::Index subObjectCount = layoutImpl->getSubObjectCount();
        this->m_objects.setCount(subObjectCount);
        auto dataSize = layoutImpl->getElementTypeLayout()->getSize();
        assert(dataSize >= 0);
        this->m_data.setCount(dataSize);
        memset(this->m_data.getBuffer(), 0, dataSize);
        return SLANG_OK;
    }

public:
    virtual SLANG_NO_THROW const void* SLANG_MCALL getRawData() override
    {
        return this->m_data.getBuffer();
    }
    virtual SLANG_NO_THROW size_t SLANG_MCALL getSize() override { return this->m_data.getCount(); }
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setData(ShaderOffset const& offset, void const* data, size_t size) override
    {
        if (!size)
            return SLANG_OK;
        if (SlangInt(offset.uniformOffset + size) > this->m_data.getCount())
            this->m_data.setCount(offset.uniformOffset + size);
        memcpy(this->m_data.getBuffer() + offset.uniformOffset, data, size);
        this->m_data.markDirty();
        markDirty();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setObject(ShaderOffset const& offset, IShaderObject* object) override
    {
        Super::setObject(offset, object);
        m_objectOffsets.add(offset);
        markDirty();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setResource(ShaderOffset const& offset, IResourceView* resourceView) override
    {
        m_resources[offset] = static_cast<ResourceViewBase*>(resourceView);
        markDirty();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setSampler(ShaderOffset const& offset, ISamplerState* sampler) override
    {
        m_samplers[offset] = static_cast<SamplerStateBase*>(sampler);
        markDirty();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL setCombinedTextureSampler(
        ShaderOffset const& offset,
        IResourceView* textureView,
        ISamplerState* sampler) override
    {
        m_samplers[offset] = static_cast<SamplerStateBase*>(sampler);
        m_resources[offset] = static_cast<ResourceViewBase*>(textureView);
        markDirty();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getCurrentVersion(ITransientResourceHeap* transientHeap, IShaderObject** outObject) override
    {
        if (!isDirty())
        {
            returnComPtr(outObject, getLastAllocatedShaderObject());
            return SLANG_OK;
        }

        Slang::RefPtr<ShaderObjectBase> object =
            allocateShaderObject(static_cast<TransientResourceHeapBase*>(transientHeap));
        SLANG_RETURN_ON_FAIL(
            object->setData(ShaderOffset(), this->m_data.getBuffer(), this->m_data.getCount()));
        for (auto res : m_resources)
            SLANG_RETURN_ON_FAIL(object->setResource(res.key, res.value));
        for (auto sampler : m_samplers)
            SLANG_RETURN_ON_FAIL(object->setSampler(sampler.key, sampler.value));
        for (auto offset : m_objectOffsets)
        {
            if (offset.bindingRangeIndex < 0)
                return SLANG_E_INVALID_ARG;
            auto layout = this->getLayout();
            if (offset.bindingRangeIndex >= layout->getBindingRangeCount())
                return SLANG_E_INVALID_ARG;
            auto bindingRange = layout->getBindingRange(offset.bindingRangeIndex);

            auto subObject =
                this->m_objects[bindingRange.subObjectIndex + offset.bindingArrayIndex];
            if (subObject)
            {
                ComPtr<IShaderObject> subObjectVersion;
                SLANG_RETURN_ON_FAIL(
                    subObject->getCurrentVersion(transientHeap, subObjectVersion.writeRef()));
                SLANG_RETURN_ON_FAIL(object->setObject(offset, subObjectVersion));
            }
        }
        m_dirty = false;
        this->m_data.m_dirty = false;
        returnComPtr(outObject, object);
        return SLANG_OK;
    }

public:
    Slang::RefPtr<ShaderObjectBase> allocateShaderObject(TransientResourceHeapBase* transientHeap)
    {
        auto& version = m_shaderObjectVersions.allocate(transientHeap);
        if (!version.object)
        {
            ComPtr<IShaderObject> shaderObject;
            SLANG_RETURN_NULL_ON_FAIL(
                this->m_device->createShaderObject(this->m_layout, shaderObject.writeRef()));
            version.object = static_cast<ShaderObjectBase*>(shaderObject.get());
        }
        return version.object;
    }
    Slang::RefPtr<ShaderObjectBase> getLastAllocatedShaderObject()
    {
        return m_shaderObjectVersions.getLastAllocation().object;
    }
};

// A proxy shader object to hold mutable shader parameters for global scope and entry-points.
class MutableRootShaderObject : public ShaderObjectBase
{
public:
    Slang::List<uint8_t> m_data;
    Slang::OrderedDictionary<ShaderOffset, Slang::RefPtr<ResourceViewBase>> m_resources;
    Slang::OrderedDictionary<ShaderOffset, Slang::RefPtr<SamplerStateBase>> m_samplers;
    Slang::OrderedDictionary<ShaderOffset, Slang::RefPtr<ShaderObjectBase>> m_objects;
    Slang::OrderedDictionary<ShaderOffset, Slang::List<slang::SpecializationArg>>
        m_specializationArgs;
    Slang::List<Slang::RefPtr<MutableRootShaderObject>> m_entryPoints;
    Slang::RefPtr<BufferResource> m_constantBufferOverride;
    slang::TypeLayoutReflection* m_elementTypeLayout;

    MutableRootShaderObject(RendererBase* device, slang::TypeLayoutReflection* entryPointLayout)
    {
        this->m_device = device;
        m_elementTypeLayout = entryPointLayout;
        m_data.setCount(entryPointLayout->getSize());
        memset(m_data.begin(), 0, m_data.getCount());
    }

    MutableRootShaderObject(RendererBase* device, Slang::RefPtr<ShaderProgramBase> program)
    {
        this->m_device = device;
        auto programLayout = program->slangGlobalScope->getLayout();
        SlangInt entryPointCount = programLayout->getEntryPointCount();
        for (SlangInt e = 0; e < entryPointCount; ++e)
        {
            auto slangEntryPoint = programLayout->getEntryPointByIndex(e);
            Slang::RefPtr<MutableRootShaderObject> entryPointObject = new MutableRootShaderObject(
                device,
                slangEntryPoint->getTypeLayout()->getElementTypeLayout());

            m_entryPoints.add(entryPointObject);
        }
        m_data.setCount(programLayout->getGlobalParamsTypeLayout()->getSize());
        memset(m_data.begin(), 0, m_data.getCount());
        m_elementTypeLayout = programLayout->getGlobalParamsTypeLayout();
    }


    virtual SLANG_NO_THROW slang::TypeLayoutReflection* SLANG_MCALL getElementTypeLayout() override
    {
        return m_elementTypeLayout;
    }

    virtual SLANG_NO_THROW ShaderObjectContainerType SLANG_MCALL getContainerType() override
    {
        return ShaderObjectContainerType::None;
    }

    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() override
    {
        return (GfxCount)m_entryPoints.getCount();
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** entryPoint) override
    {
        returnComPtr(entryPoint, m_entryPoints[index]);
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setData(ShaderOffset const& offset, void const* data, Size size) override
    {
        auto newSize = Slang::Index(size + offset.uniformOffset);
        if (newSize > m_data.getCount())
            m_data.setCount((Slang::Index)newSize);
        memcpy(m_data.begin() + offset.uniformOffset, data, size);
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getObject(ShaderOffset const& offset, IShaderObject** object) override
    {
        *object = nullptr;

        Slang::RefPtr<ShaderObjectBase> subObject;
        if (m_objects.tryGetValue(offset, subObject))
        {
            returnComPtr(object, subObject);
        }
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setObject(ShaderOffset const& offset, IShaderObject* object) override
    {
        m_objects[offset] = static_cast<ShaderObjectBase*>(object);
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setResource(ShaderOffset const& offset, IResourceView* resourceView) override
    {
        m_resources[offset] = static_cast<ResourceViewBase*>(resourceView);
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setSampler(ShaderOffset const& offset, ISamplerState* sampler) override
    {
        m_samplers[offset] = static_cast<SamplerStateBase*>(sampler);
        return SLANG_OK;
    }
    virtual SLANG_NO_THROW Result SLANG_MCALL setCombinedTextureSampler(
        ShaderOffset const& offset,
        IResourceView* textureView,
        ISamplerState* sampler) override
    {
        m_resources[offset] = static_cast<ResourceViewBase*>(textureView);
        m_samplers[offset] = static_cast<SamplerStateBase*>(sampler);
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL setSpecializationArgs(
        ShaderOffset const& offset,
        const slang::SpecializationArg* args,
        GfxCount count) override
    {
        Slang::List<slang::SpecializationArg> specArgs;
        specArgs.addRange(args, count);
        m_specializationArgs[offset] = specArgs;
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getCurrentVersion(ITransientResourceHeap* transientHeap, IShaderObject** outObject) override
    {
        return SLANG_FAIL;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    copyFrom(IShaderObject* other, ITransientResourceHeap* transientHeap) override
    {
        auto otherObject = static_cast<MutableRootShaderObject*>(other);
        *this = *otherObject;
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW const void* SLANG_MCALL getRawData() override { return m_data.begin(); }

    virtual SLANG_NO_THROW Size SLANG_MCALL getSize() override { return (Size)m_data.getCount(); }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    setConstantBufferOverride(IBufferResource* constantBuffer) override
    {
        m_constantBufferOverride = static_cast<BufferResource*>(constantBuffer);
        return SLANG_OK;
    }

    virtual Result collectSpecializationArgs(ExtendedShaderObjectTypeList& args) override
    {
        SLANG_UNUSED(args);
        return SLANG_OK;
    }
};

} // namespace gfx
