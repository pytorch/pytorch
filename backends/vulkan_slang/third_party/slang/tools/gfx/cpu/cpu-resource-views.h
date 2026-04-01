// cpu-resource-views.h
#pragma once
#include "cpu-base.h"
#include "cpu-buffer.h"
#include "cpu-texture.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

class ResourceViewImpl : public ResourceViewBase
{
public:
    enum class Kind
    {
        Buffer,
        Texture,
    };
    Kind getViewKind() const { return m_kind; }
    Desc const& getDesc() const { return m_desc; }

protected:
    ResourceViewImpl(Kind kind, Desc const& desc);

private:
    Kind m_kind;
};

class BufferResourceViewImpl : public ResourceViewImpl
{
public:
    BufferResourceViewImpl(Desc const& desc, BufferResourceImpl* buffer)
        : ResourceViewImpl(Kind::Buffer, desc), m_buffer(buffer)
    {
    }

    BufferResourceImpl* getBuffer() const;

private:
    RefPtr<BufferResourceImpl> m_buffer;
};

class TextureResourceViewImpl : public ResourceViewImpl, public slang_prelude::IRWTexture
{
public:
    TextureResourceViewImpl(Desc const& desc, TextureResourceImpl* texture)
        : ResourceViewImpl(Kind::Texture, desc), m_texture(texture)
    {
    }

    TextureResourceImpl* getTexture() const;

    //
    // ITexture interface
    //

    slang_prelude::TextureDimensions GetDimensions(int mipLevel = -1) SLANG_OVERRIDE;

    void Load(const int32_t* texelCoords, void* outData, size_t dataSize) SLANG_OVERRIDE;

    void Sample(
        slang_prelude::SamplerState samplerState,
        const float* coords,
        void* outData,
        size_t dataSize) SLANG_OVERRIDE;

    void SampleLevel(
        slang_prelude::SamplerState samplerState,
        const float* coords,
        float level,
        void* outData,
        size_t dataSize) SLANG_OVERRIDE;

    //
    // IRWTexture interface
    //

    void* refAt(const uint32_t* texelCoords) SLANG_OVERRIDE;

private:
    RefPtr<TextureResourceImpl> m_texture;

    void* _getTexelPtr(int32_t const* texelCoords);
};

} // namespace cpu
} // namespace gfx
