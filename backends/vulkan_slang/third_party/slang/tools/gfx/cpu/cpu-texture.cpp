// cpu-texture.cpp
#include "cpu-texture.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

static CPUTextureBaseShapeInfo const* _getBaseShapeInfo(ITextureResource::Type baseShape)
{
    return &kCPUTextureBaseShapeInfos[(int)baseShape];
}

template<int N>
void _unpackFloatTexel(void const* texelData, void* outData, size_t outSize)
{
    auto input = (float const*)texelData;

    float temp[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < N; ++i)
        temp[i] = input[i];

    memcpy(outData, temp, outSize);
}

template<int N>
void _unpackFloat16Texel(void const* texelData, void* outData, size_t outSize)
{
    auto input = (int16_t const*)texelData;

    float temp[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < N; ++i)
        temp[i] = HalfToFloat(input[i]);

    memcpy(outData, temp, outSize);
}

static inline float _unpackUnorm8Value(uint8_t value)
{
    return value / 255.0f;
}

template<int N>
void _unpackUnorm8Texel(void const* texelData, void* outData, size_t outSize)
{
    auto input = (uint8_t const*)texelData;

    float temp[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < N; ++i)
        temp[i] = _unpackUnorm8Value(input[i]);

    memcpy(outData, temp, outSize);
}

void _unpackUnormBGRA8Texel(void const* texelData, void* outData, size_t outSize)
{
    auto input = (uint8_t const*)texelData;

    float temp[4];
    temp[0] = _unpackUnorm8Value(input[2]);
    temp[1] = _unpackUnorm8Value(input[1]);
    temp[2] = _unpackUnorm8Value(input[0]);
    temp[3] = _unpackUnorm8Value(input[3]);

    memcpy(outData, temp, outSize);
}

template<int N>
void _unpackUInt16Texel(void const* texelData, void* outData, size_t outSize)
{
    auto input = (uint16_t const*)texelData;

    uint32_t temp[4] = {0, 0, 0, 0};
    for (int i = 0; i < N; ++i)
        temp[i] = input[i];

    memcpy(outData, temp, outSize);
}

template<int N>
void _unpackUInt32Texel(void const* texelData, void* outData, size_t outSize)
{
    auto input = (uint32_t const*)texelData;

    uint32_t temp[4] = {0, 0, 0, 0};
    for (int i = 0; i < N; ++i)
        temp[i] = input[i];

    memcpy(outData, temp, outSize);
}

TextureResourceImpl::~TextureResourceImpl()
{
    free(m_data);
}

Result TextureResourceImpl::init(ITextureResource::SubresourceData const* initData)
{
    auto desc = m_desc;

    // The format of the texture will determine the
    // size of the texels we allocate.
    //
    // TODO: Compressed formats usually work in terms
    // of a fixed block size, so that we cannot actually
    // compute a simple `texelSize` like this. Instead
    // we should be computing a `blockSize` and then
    // a `blockExtents` value that gives the extent
    // in texels of each block. For uncompressed formats
    // the block extents would be 1 along each axis.
    //
    auto format = desc.format;
    FormatInfo texelInfo;
    gfxGetFormatInfo(format, &texelInfo);
    uint32_t texelSize = uint32_t(texelInfo.blockSizeInBytes / texelInfo.pixelsPerBlock);
    m_texelSize = texelSize;

    int32_t formatBlockSize[kMaxRank] = {1, 1, 1};

    auto baseShapeInfo = _getBaseShapeInfo(desc.type);
    m_baseShape = baseShapeInfo;
    if (!baseShapeInfo)
        return SLANG_FAIL;

    auto formatInfo = _getFormatInfo(desc.format);
    m_formatInfo = formatInfo;
    if (!formatInfo)
        return SLANG_FAIL;

    int32_t rank = baseShapeInfo->rank;
    int32_t effectiveArrayElementCount = desc.arraySize ? desc.arraySize : 1;
    effectiveArrayElementCount *= baseShapeInfo->implicitArrayElementCount;
    m_effectiveArrayElementCount = effectiveArrayElementCount;

    int32_t extents[kMaxRank];
    extents[0] = desc.size.width;
    extents[1] = desc.size.height;
    extents[2] = desc.size.depth;

    for (int32_t axis = rank; axis < kMaxRank; ++axis)
        extents[axis] = 1;

    int32_t levelCount = desc.numMipLevels;

    m_mipLevels.setCount(levelCount);

    int64_t totalDataSize = 0;
    for (int32_t levelIndex = 0; levelIndex < levelCount; ++levelIndex)
    {
        auto& level = m_mipLevels[levelIndex];

        for (int32_t axis = 0; axis < kMaxRank; ++axis)
        {
            int32_t extent = extents[axis] >> levelIndex;
            if (extent < 1)
                extent = 1;
            level.extents[axis] = extent;
        }

        level.strides[0] = texelSize;
        for (int32_t axis = 1; axis < kMaxRank + 1; ++axis)
        {
            level.strides[axis] = level.strides[axis - 1] * level.extents[axis - 1];
        }

        int64_t levelDataSize = texelSize;
        levelDataSize *= effectiveArrayElementCount;
        for (int32_t axis = 0; axis < rank; ++axis)
            levelDataSize *= int64_t(level.extents[axis]);

        level.offset = totalDataSize;
        totalDataSize += levelDataSize;
    }

    void* textureData = malloc((size_t)totalDataSize);
    m_data = textureData;

    if (initData)
    {
        int32_t subResourceCounter = 0;
        for (int32_t arrayElementIndex = 0; arrayElementIndex < effectiveArrayElementCount;
             ++arrayElementIndex)
        {
            for (int32_t mipLevel = 0; mipLevel < m_desc.numMipLevels; ++mipLevel)
            {
                int32_t subResourceIndex = subResourceCounter++;

                auto dstRowStride = m_mipLevels[mipLevel].strides[1];
                auto dstLayerStride = m_mipLevels[mipLevel].strides[2];
                auto dstArrayStride = m_mipLevels[mipLevel].strides[3];

                auto textureRowSize = m_mipLevels[mipLevel].extents[0] * texelSize;

                auto rowCount = m_mipLevels[mipLevel].extents[1];
                auto depthLayerCount = m_mipLevels[mipLevel].extents[2];

                auto& srcImage = initData[subResourceIndex];
                ptrdiff_t srcRowStride = ptrdiff_t(srcImage.strideY);
                ptrdiff_t srcLayerStride = ptrdiff_t(srcImage.strideZ);

                char* dstLevel = (char*)textureData + m_mipLevels[mipLevel].offset;
                char* dstImage = dstLevel + dstArrayStride * arrayElementIndex;

                const char* srcLayer = (const char*)srcImage.data;
                char* dstLayer = dstImage;

                for (int32_t depthLayer = 0; depthLayer < depthLayerCount; ++depthLayer)
                {
                    const char* srcRow = srcLayer;
                    char* dstRow = dstLayer;

                    for (int32_t row = 0; row < rowCount; ++row)
                    {
                        memcpy(dstRow, srcRow, textureRowSize);

                        srcRow += srcRowStride;
                        dstRow += dstRowStride;
                    }

                    srcLayer += srcLayerStride;
                    dstLayer += dstLayerStride;
                }
            }
        }
    }

    return SLANG_OK;
}

} // namespace cpu
} // namespace gfx
