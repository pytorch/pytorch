// metal-texture.cpp
#include "metal-texture.h"

#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

TextureResourceImpl::TextureResourceImpl(const Desc& desc, DeviceImpl* device)
    : Parent(desc), m_device(device)
{
}

TextureResourceImpl::~TextureResourceImpl() {}

Result TextureResourceImpl::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Metal;
    outHandle->handleValue = reinterpret_cast<intptr_t>(m_texture.get());
    return SLANG_OK;
}

Result TextureResourceImpl::getSharedHandle(InteropHandle* outHandle)
{
    return SLANG_E_NOT_AVAILABLE;
}

Result TextureResourceImpl::setDebugName(const char* name)
{
    Parent::setDebugName(name);
    m_texture->setLabel(MetalUtil::createString(name).get());
    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
