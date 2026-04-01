#include "gfx-test-texture-util.h"

#include "gfx-test-util.h"
#include "slang-com-ptr.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define GFX_ENABLE_RENDERDOC_INTEGRATION 0

#if GFX_ENABLE_RENDERDOC_INTEGRATION
#include "external/renderdoc_app.h"

#include <windows.h>
#endif

using namespace Slang;
using namespace gfx;

namespace gfx_test
{
TextureAspect getTextureAspect(Format format)
{
    switch (format)
    {
    case Format::D16_UNORM:
    case Format::D32_FLOAT:
        return TextureAspect::Depth;
    default:
        return TextureAspect::Color;
    }
}

Size getTexelSize(Format format)
{
    FormatInfo info;
    GFX_CHECK_CALL_ABORT(gfxGetFormatInfo(format, &info));
    return info.blockSizeInBytes / info.pixelsPerBlock;
}

GfxIndex getSubresourceIndex(GfxIndex mipLevel, GfxCount mipLevelCount, GfxIndex baseArrayLayer)
{
    return baseArrayLayer * mipLevelCount + mipLevel;
}

RefPtr<ValidationTextureFormatBase> getValidationTextureFormat(Format format)
{
    switch (format)
    {
    case Format::R32G32B32A32_TYPELESS:
        return new ValidationTextureFormat<uint32_t>(4);
    case Format::R32G32B32_TYPELESS:
        return new ValidationTextureFormat<uint32_t>(3);
    case Format::R32G32_TYPELESS:
        return new ValidationTextureFormat<uint32_t>(2);
    case Format::R32_TYPELESS:
        return new ValidationTextureFormat<uint32_t>(1);

    case Format::R16G16B16A16_TYPELESS:
        return new ValidationTextureFormat<uint16_t>(4);
    case Format::R16G16_TYPELESS:
        return new ValidationTextureFormat<uint16_t>(2);
    case Format::R16_TYPELESS:
        return new ValidationTextureFormat<uint16_t>(1);

    case Format::R8G8B8A8_TYPELESS:
        return new ValidationTextureFormat<uint8_t>(4);
    case Format::R8G8_TYPELESS:
        return new ValidationTextureFormat<uint8_t>(2);
    case Format::R8_TYPELESS:
        return new ValidationTextureFormat<uint8_t>(1);
    case Format::B8G8R8A8_TYPELESS:
        return new ValidationTextureFormat<uint8_t>(4);

    case Format::R32G32B32A32_FLOAT:
        return new ValidationTextureFormat<float>(4);
    case Format::R32G32B32_FLOAT:
        return new ValidationTextureFormat<float>(3);
    case Format::R32G32_FLOAT:
        return new ValidationTextureFormat<float>(2);
    case Format::R32_FLOAT:
        return new ValidationTextureFormat<float>(1);

    case Format::R16G16B16A16_FLOAT:
        return new ValidationTextureFormat<uint16_t>(4);
    case Format::R16G16_FLOAT:
        return new ValidationTextureFormat<uint16_t>(2);
    case Format::R16_FLOAT:
        return new ValidationTextureFormat<uint16_t>(1);

    case Format::R64_UINT:
        return new ValidationTextureFormat<uint64_t>(1);

    case Format::R32G32B32A32_UINT:
        return new ValidationTextureFormat<uint32_t>(4);
    case Format::R32G32B32_UINT:
        return new ValidationTextureFormat<uint32_t>(3);
    case Format::R32G32_UINT:
        return new ValidationTextureFormat<uint32_t>(2);
    case Format::R32_UINT:
        return new ValidationTextureFormat<uint32_t>(1);

    case Format::R16G16B16A16_UINT:
        return new ValidationTextureFormat<uint16_t>(4);
    case Format::R16G16_UINT:
        return new ValidationTextureFormat<uint16_t>(2);
    case Format::R16_UINT:
        return new ValidationTextureFormat<uint16_t>(1);

    case Format::R8G8B8A8_UINT:
        return new ValidationTextureFormat<uint8_t>(4);
    case Format::R8G8_UINT:
        return new ValidationTextureFormat<uint8_t>(2);
    case Format::R8_UINT:
        return new ValidationTextureFormat<uint8_t>(1);

    case Format::R64_SINT:
        return new ValidationTextureFormat<int64_t>(1);

    case Format::R32G32B32A32_SINT:
        return new ValidationTextureFormat<int32_t>(4);
    case Format::R32G32B32_SINT:
        return new ValidationTextureFormat<int32_t>(3);
    case Format::R32G32_SINT:
        return new ValidationTextureFormat<int32_t>(2);
    case Format::R32_SINT:
        return new ValidationTextureFormat<int32_t>(1);

    case Format::R16G16B16A16_SINT:
        return new ValidationTextureFormat<int16_t>(4);
    case Format::R16G16_SINT:
        return new ValidationTextureFormat<int16_t>(2);
    case Format::R16_SINT:
        return new ValidationTextureFormat<int16_t>(1);

    case Format::R8G8B8A8_SINT:
        return new ValidationTextureFormat<int8_t>(4);
    case Format::R8G8_SINT:
        return new ValidationTextureFormat<int8_t>(2);
    case Format::R8_SINT:
        return new ValidationTextureFormat<int8_t>(1);

    case Format::R16G16B16A16_UNORM:
        return new ValidationTextureFormat<uint16_t>(4);
    case Format::R16G16_UNORM:
        return new ValidationTextureFormat<uint16_t>(2);
    case Format::R16_UNORM:
        return new ValidationTextureFormat<uint16_t>(1);

    case Format::R8G8B8A8_UNORM:
        return new ValidationTextureFormat<uint8_t>(4);
    case Format::R8G8B8A8_UNORM_SRGB:
        return new ValidationTextureFormat<uint8_t>(4);
    case Format::R8G8_UNORM:
        return new ValidationTextureFormat<uint8_t>(2);
    case Format::R8_UNORM:
        return new ValidationTextureFormat<uint8_t>(1);
    case Format::B8G8R8A8_UNORM:
        return new ValidationTextureFormat<uint8_t>(4);
    case Format::B8G8R8A8_UNORM_SRGB:
        return new ValidationTextureFormat<uint8_t>(4);
    case Format::B8G8R8X8_UNORM:
        return new ValidationTextureFormat<uint8_t>(3);
    case Format::B8G8R8X8_UNORM_SRGB:
        return new ValidationTextureFormat<uint8_t>(3);

    case Format::R16G16B16A16_SNORM:
        return new ValidationTextureFormat<int16_t>(4);
    case Format::R16G16_SNORM:
        return new ValidationTextureFormat<int16_t>(2);
    case Format::R16_SNORM:
        return new ValidationTextureFormat<int16_t>(1);

    case Format::R8G8B8A8_SNORM:
        return new ValidationTextureFormat<int8_t>(4);
    case Format::R8G8_SNORM:
        return new ValidationTextureFormat<int8_t>(2);
    case Format::R8_SNORM:
        return new ValidationTextureFormat<int8_t>(1);

    case Format::D32_FLOAT:
        return new ValidationTextureFormat<float>(1);
    case Format::D16_UNORM:
        return new ValidationTextureFormat<uint16_t>(1);

    case Format::B4G4R4A4_UNORM:
        return new PackedValidationTextureFormat<uint16_t>(4, 4, 4, 4);
    case Format::B5G6R5_UNORM:
        return new PackedValidationTextureFormat<uint16_t>(5, 6, 5, 0);
    case Format::B5G5R5A1_UNORM:
        return new PackedValidationTextureFormat<uint16_t>(5, 5, 5, 1);

    case Format::R9G9B9E5_SHAREDEXP:
        return new ValidationTextureFormat<uint32_t>(1);
    case Format::R10G10B10A2_TYPELESS:
        return new PackedValidationTextureFormat<uint32_t>(10, 10, 10, 2);
    case Format::R10G10B10A2_UNORM:
        return new PackedValidationTextureFormat<uint32_t>(10, 10, 10, 2);
    case Format::R10G10B10A2_UINT:
        return new PackedValidationTextureFormat<uint32_t>(10, 10, 10, 2);
    case Format::R11G11B10_FLOAT:
        return new PackedValidationTextureFormat<uint32_t>(11, 11, 10, 0);

        // TODO: Add testing support for BC formats
        //                     BC1_UNORM,
        //                     BC1_UNORM_SRGB,
        //                     BC2_UNORM,
        //                     BC2_UNORM_SRGB,
        //                     BC3_UNORM,
        //                     BC3_UNORM_SRGB,
        //                     BC4_UNORM,
        //                     BC4_SNORM,
        //                     BC5_UNORM,
        //                     BC5_SNORM,
        //                     BC6H_UF16,
        //                     BC6H_SF16,
        //                     BC7_UNORM,
        //                     BC7_UNORM_SRGB,
    default:
        return nullptr;
    }
}

void generateTextureData(RefPtr<TextureInfo> texture, ValidationTextureFormatBase* validationFormat)
{
    auto extents = texture->extents;
    auto arrayLayers = texture->arrayLayerCount;
    auto mipLevels = texture->mipLevelCount;
    auto texelSize = getTexelSize(texture->format);

    for (GfxIndex layer = 0; layer < arrayLayers; ++layer)
    {
        for (GfxIndex mip = 0; mip < mipLevels; ++mip)
        {
            RefPtr<ValidationTextureData> subresource = new ValidationTextureData();

            auto mipWidth = Math::Max(extents.width >> mip, 1);
            auto mipHeight = Math::Max(extents.height >> mip, 1);
            auto mipDepth = Math::Max(extents.depth >> mip, 1);
            auto mipSize = mipWidth * mipHeight * mipDepth * texelSize;
            subresource->textureData = malloc(mipSize);
            SLANG_CHECK_ABORT(subresource->textureData);

            subresource->extents.width = mipWidth;
            subresource->extents.height = mipHeight;
            subresource->extents.depth = mipDepth;
            subresource->strides.x = texelSize;
            subresource->strides.y = mipWidth * texelSize;
            subresource->strides.z = mipHeight * subresource->strides.y;
            texture->subresourceObjects.add(subresource);

            for (int z = 0; z < mipDepth; ++z)
            {
                for (int y = 0; y < mipHeight; ++y)
                {
                    for (int x = 0; x < mipWidth; ++x)
                    {
                        auto texel = subresource->getBlockAt(x, y, z);
                        validationFormat->initializeTexel(texel, x, y, z, mip, layer);
                    }
                }
            }

            ITextureResource::SubresourceData subData = {};
            subData.data = subresource->textureData;
            subData.strideY = subresource->strides.y;
            subData.strideZ = subresource->strides.z;
            texture->subresourceDatas.add(subData);
        }
    }
}

List<uint8_t> removePadding(
    ISlangBlob* pixels,
    GfxCount width,
    GfxCount height,
    Size rowPitch,
    Size pixelSize)
{
    List<uint8_t> buffer;
    buffer.setCount(height * rowPitch);
    for (GfxIndex i = 0; i < height; ++i)
    {
        Offset srcOffset = i * rowPitch;
        Offset dstOffset = i * width * pixelSize;
        memcpy(
            buffer.getBuffer() + dstOffset,
            (char*)pixels->getBufferPointer() + srcOffset,
            width * pixelSize);
    }

    return buffer;
}

Slang::Result writeImage(const char* filename, ISlangBlob* pixels, uint32_t width, uint32_t height)
{
    int stbResult = stbi_write_hdr(filename, width, height, 4, (float*)pixels->getBufferPointer());

    return stbResult ? SLANG_OK : SLANG_FAIL;
}

Slang::Result writeImage(
    const char* filename,
    ISlangBlob* pixels,
    uint32_t width,
    uint32_t height,
    uint32_t rowPitch,
    uint32_t pixelSize)
{
    if (rowPitch == width * pixelSize)
        return writeImage(filename, pixels, width, height);

    List<uint8_t> buffer = removePadding(pixels, width, height, rowPitch, pixelSize);

    int stbResult = stbi_write_hdr(filename, width, height, 4, (float*)buffer.getBuffer());

    return stbResult ? SLANG_OK : SLANG_FAIL;
}
} // namespace gfx_test
