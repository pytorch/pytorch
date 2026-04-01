// debug-shader-object.cpp
#include "debug-shader-object.h"

#include "debug-helper-functions.h"
#include "debug-resource-views.h"
#include "debug-sampler-state.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

ShaderObjectContainerType DebugShaderObject::getContainerType()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getContainerType();
}

void DebugShaderObject::checkCompleteness()
{
    auto layout = baseObject->getElementTypeLayout();
    for (Index i = 0; i < layout->getBindingRangeCount(); i++)
    {
        if (layout->getBindingRangeBindingCount(i) != 0)
        {
            if (!m_initializedBindingRanges.contains(i))
            {
                auto var = layout->getBindingRangeLeafVariable(i);
                GFX_DIAGNOSE_ERROR_FORMAT(
                    "shader parameter '%s' is not initialized in the shader object of type '%s'.",
                    var->getName(),
                    m_slangType->getName());
            }
        }
    }
}

slang::TypeLayoutReflection* DebugShaderObject::getElementTypeLayout()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getElementTypeLayout();
}

GfxCount DebugShaderObject::getEntryPointCount()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getEntryPointCount();
}

Result DebugShaderObject::getEntryPoint(GfxIndex index, IShaderObject** entryPoint)
{
    SLANG_GFX_API_FUNC;
    if (m_entryPoints.getCount() == 0)
    {
        for (GfxIndex i = 0; i < getEntryPointCount(); i++)
        {
            RefPtr<DebugShaderObject> entryPointObj = new DebugShaderObject();
            SLANG_RETURN_ON_FAIL(
                baseObject->getEntryPoint(i, entryPointObj->baseObject.writeRef()));
            m_entryPoints.add(entryPointObj);
        }
    }
    if (index > (GfxCount)m_entryPoints.getCount())
    {
        GFX_DIAGNOSE_ERROR("`index` must not exceed `entryPointCount`.");
        return SLANG_FAIL;
    }
    returnComPtr(entryPoint, m_entryPoints[index]);
    return SLANG_OK;
}

Result DebugShaderObject::setData(ShaderOffset const& offset, void const* data, Size size)
{
    SLANG_GFX_API_FUNC;
    return baseObject->setData(offset, data, size);
}

Result DebugShaderObject::getObject(ShaderOffset const& offset, IShaderObject** object)
{
    SLANG_GFX_API_FUNC;

    ComPtr<IShaderObject> innerObject;
    auto resultCode = baseObject->getObject(offset, innerObject.writeRef());
    SLANG_RETURN_ON_FAIL(resultCode);
    RefPtr<DebugShaderObject> debugShaderObject;
    if (m_objects.tryGetValue(ShaderOffsetKey{offset}, debugShaderObject))
    {
        if (debugShaderObject->baseObject == innerObject)
        {
            returnComPtr(object, debugShaderObject);
            return resultCode;
        }
    }
    debugShaderObject = new DebugShaderObject();
    debugShaderObject->baseObject = innerObject;
    debugShaderObject->m_typeName = innerObject->getElementTypeLayout()->getName();
    m_objects[ShaderOffsetKey{offset}] = debugShaderObject;
    returnComPtr(object, debugShaderObject);
    return resultCode;
}

Result DebugShaderObject::setObject(ShaderOffset const& offset, IShaderObject* object)
{
    SLANG_GFX_API_FUNC;
    auto objectImpl = getDebugObj(object);
    m_objects[ShaderOffsetKey{offset}] = objectImpl;
    m_initializedBindingRanges.add(offset.bindingRangeIndex);
    objectImpl->checkCompleteness();
    return baseObject->setObject(offset, getInnerObj(object));
}

Result DebugShaderObject::setResource(ShaderOffset const& offset, IResourceView* resourceView)
{
    SLANG_GFX_API_FUNC;
    auto viewImpl = getDebugObj(resourceView);
    m_resources[ShaderOffsetKey{offset}] = viewImpl;
    m_initializedBindingRanges.add(offset.bindingRangeIndex);
    return baseObject->setResource(offset, getInnerObj(resourceView));
}

Result DebugShaderObject::setSampler(ShaderOffset const& offset, ISamplerState* sampler)
{
    SLANG_GFX_API_FUNC;
    auto samplerImpl = getDebugObj(sampler);
    m_samplers[ShaderOffsetKey{offset}] = samplerImpl;
    m_initializedBindingRanges.add(offset.bindingRangeIndex);
    return baseObject->setSampler(offset, getInnerObj(sampler));
}

Result DebugShaderObject::setCombinedTextureSampler(
    ShaderOffset const& offset,
    IResourceView* textureView,
    ISamplerState* sampler)
{
    SLANG_GFX_API_FUNC;
    auto samplerImpl = getDebugObj(sampler);
    m_samplers[ShaderOffsetKey{offset}] = samplerImpl;
    auto viewImpl = getDebugObj(textureView);
    m_resources[ShaderOffsetKey{offset}] = viewImpl;
    m_initializedBindingRanges.add(offset.bindingRangeIndex);
    return baseObject->setCombinedTextureSampler(
        offset,
        getInnerObj(viewImpl),
        getInnerObj(sampler));
}

Result DebugShaderObject::setSpecializationArgs(
    ShaderOffset const& offset,
    const slang::SpecializationArg* args,
    GfxCount count)
{
    SLANG_GFX_API_FUNC;
    return baseObject->setSpecializationArgs(offset, args, count);
}

Result DebugShaderObject::getCurrentVersion(
    ITransientResourceHeap* transientHeap,
    IShaderObject** outObject)
{
    SLANG_GFX_API_FUNC;
    ComPtr<IShaderObject> innerObject;
    SLANG_RETURN_ON_FAIL(
        baseObject->getCurrentVersion(getInnerObj(transientHeap), innerObject.writeRef()));
    RefPtr<DebugShaderObject> debugShaderObject = new DebugShaderObject();
    debugShaderObject->baseObject = innerObject;
    debugShaderObject->m_typeName = innerObject->getElementTypeLayout()->getName();
    returnComPtr(outObject, debugShaderObject);
    return SLANG_OK;
}

const void* DebugShaderObject::getRawData()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getRawData();
}

size_t DebugShaderObject::getSize()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getSize();
}

Result DebugShaderObject::setConstantBufferOverride(IBufferResource* constantBuffer)
{
    SLANG_GFX_API_FUNC;
    return baseObject->setConstantBufferOverride(getInnerObj(constantBuffer));
}

Result DebugRootShaderObject::setSpecializationArgs(
    ShaderOffset const& offset,
    const slang::SpecializationArg* args,
    GfxCount count)
{
    SLANG_GFX_API_FUNC;

    return baseObject->setSpecializationArgs(offset, args, count);
}

void DebugRootShaderObject::reset()
{
    m_entryPoints.clear();
    m_objects.clear();
    m_resources.clear();
    m_samplers.clear();
    baseObject.detach();
}

} // namespace debug
} // namespace gfx
