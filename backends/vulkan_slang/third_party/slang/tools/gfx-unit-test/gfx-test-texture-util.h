#pragma once

#include "core/slang-basic.h"
#include "core/slang-render-api-util.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;
using namespace gfx;

namespace gfx_test
{
struct Strides
{
    Size x;
    Size y;
    Size z;
};

struct ValidationTextureFormatBase : RefObject
{
    virtual void validateBlocksEqual(const void* actual, const void* expected) = 0;

    virtual void initializeTexel(
        void* texel,
        GfxIndex x,
        GfxIndex y,
        GfxIndex z,
        GfxIndex mipLevel,
        GfxIndex arrayLayer) = 0;
};

template<typename T>
struct ValidationTextureFormat : ValidationTextureFormatBase
{
    int componentCount;

    ValidationTextureFormat(int componentCount)
        : componentCount(componentCount){};

    virtual void validateBlocksEqual(const void* actual, const void* expected) override
    {
        auto a = (const T*)actual;
        auto e = (const T*)expected;

        for (Int i = 0; i < componentCount; ++i)
        {
            SLANG_CHECK(a[i] == e[i]);
        }
    }

    virtual void initializeTexel(
        void* texel,
        GfxIndex x,
        GfxIndex y,
        GfxIndex z,
        GfxIndex mipLevel,
        GfxIndex arrayLayer) override
    {
        auto temp = (T*)texel;

        switch (componentCount)
        {
        case 1:
            temp[0] = T(x + y + z + mipLevel + arrayLayer);
            break;
        case 2:
            temp[0] = T(x + z + arrayLayer);
            temp[1] = T(y + mipLevel);
            break;
        case 3:
            temp[0] = T(x + mipLevel);
            temp[1] = T(y + arrayLayer);
            temp[2] = T(z);
            break;
        case 4:
            temp[0] = T(x + arrayLayer);
            temp[1] = (T)y;
            temp[2] = (T)z;
            temp[3] = (T)mipLevel;
            break;
        default:
            assert(!"component count should be no greater than 4");
            SLANG_CHECK_ABORT(false);
        }
    }
};

template<typename T>
struct PackedValidationTextureFormat : ValidationTextureFormatBase
{
    int rBits;
    int gBits;
    int bBits;
    int aBits;

    PackedValidationTextureFormat(int rBits, int gBits, int bBits, int aBits)
        : rBits(rBits), gBits(gBits), bBits(bBits), aBits(aBits){};

    virtual void validateBlocksEqual(const void* actual, const void* expected) override
    {
        T a[4];
        T e[4];
        unpackTexel(*(const T*)actual, a);
        unpackTexel(*(const T*)expected, e);

        for (Int i = 0; i < 4; ++i)
        {
            SLANG_CHECK(a[i] == e[i]);
        }
    }

    virtual void initializeTexel(
        void* texel,
        GfxIndex x,
        GfxIndex y,
        GfxIndex z,
        GfxIndex mipLevel,
        GfxIndex arrayLayer) override
    {
        T temp = 0;

        // The only formats which currently use this have either 3 or 4 channels. TODO: BC formats?
        if (aBits == 0)
        {
            temp |= z;
            temp <<= gBits;
            temp |= (y + arrayLayer);
            temp <<= rBits;
            temp |= (x + mipLevel);
        }
        else
        {
            temp |= mipLevel;
            temp <<= bBits;
            temp |= z;
            temp <<= gBits;
            temp |= y;
            temp <<= rBits;
            temp |= (x + arrayLayer);
        }

        *(T*)texel = temp;
    }

    void unpackTexel(T texel, T* outComponents)
    {
        outComponents[0] = texel & ((1 << rBits) - 1);
        texel >>= rBits;

        outComponents[1] = texel & ((1 << gBits) - 1);
        texel >>= gBits;

        outComponents[2] = texel & ((1 << bBits) - 1);
        texel >>= bBits;

        outComponents[3] = texel & ((1 << aBits) - 1);
        texel >>= aBits;
    }
};

// Struct containing texture data and information for a specific subresource.
struct ValidationTextureData : RefObject
{
    const void* textureData;
    ITextureResource::Extents extents;
    Strides strides;

    void* getBlockAt(GfxIndex x, GfxIndex y, GfxIndex z)
    {
        assert(x >= 0 && x < extents.width);
        assert(y >= 0 && y < extents.height);
        assert(z >= 0 && z < extents.depth);

        char* layerData = (char*)textureData + z * strides.z;
        char* rowData = layerData + y * strides.y;
        return rowData + x * strides.x;
    }
};

// Struct containing relevant information for a texture, including a list of its subresources
// and all relevant information for each subresource.
struct TextureInfo : RefObject
{
    Format format;
    ITextureResource::Type textureType;

    ITextureResource::Extents extents;
    GfxCount mipLevelCount;
    GfxCount arrayLayerCount;

    List<RefPtr<ValidationTextureData>> subresourceObjects;
    List<ITextureResource::SubresourceData> subresourceDatas;
};

TextureAspect getTextureAspect(Format format);
Size getTexelSize(Format format);
GfxIndex getSubresourceIndex(GfxIndex mipLevel, GfxCount mipLevelCount, GfxIndex baseArrayLayer);
RefPtr<ValidationTextureFormatBase> getValidationTextureFormat(Format format);
void generateTextureData(
    RefPtr<TextureInfo> texture,
    ValidationTextureFormatBase* validationFormat);

List<uint8_t> removePadding(
    ISlangBlob* pixels,
    GfxCount width,
    GfxCount height,
    Size rowPitch,
    Size pixelSize);
Slang::Result writeImage(const char* filename, ISlangBlob* pixels, uint32_t width, uint32_t height);
Slang::Result writeImage(
    const char* filename,
    ISlangBlob* pixels,
    uint32_t width,
    uint32_t height,
    uint32_t rowPitch,
    uint32_t pixelSize);
} // namespace gfx_test
