#pragma once

#include "core/slang-math.h"
#include "slang-gfx.h"

namespace gfx
{

inline int calcMipSize(int size, int level)
{
    size = size >> level;
    return size > 0 ? size : 1;
}

inline ITextureResource::Extents calcMipSize(ITextureResource::Extents size, int mipLevel)
{
    ITextureResource::Extents rs;
    rs.width = calcMipSize(size.width, mipLevel);
    rs.height = calcMipSize(size.height, mipLevel);
    rs.depth = calcMipSize(size.depth, mipLevel);
    return rs;
}

/// Calculate the effective array size - in essence the amount if mip map sets needed.
/// In practice takes into account if the arraySize is 0 (it's not an array, but it will still have
/// at least one mip set) and if the type is a cubemap (multiplies the amount of mip sets by 6)
inline int calcEffectiveArraySize(const ITextureResource::Desc& desc)
{
    const int arrSize = (desc.arraySize > 0) ? desc.arraySize : 1;

    switch (desc.type)
    {
    case IResource::Type::Texture1D: // fallthru
    case IResource::Type::Texture2D:
        {
            return arrSize;
        }
    case IResource::Type::TextureCube:
        return arrSize * 6;
    case IResource::Type::Texture3D:
        return 1;
    default:
        return 0;
    }
}

/// Given the type works out the maximum dimension size
inline int calcMaxDimension(ITextureResource::Extents size, IResource::Type type)
{
    switch (type)
    {
    case IResource::Type::Texture1D:
        return size.width;
    case IResource::Type::Texture3D:
        return Slang::Math::Max(Slang::Math::Max(size.width, size.height), size.depth);
    case IResource::Type::TextureCube: // fallthru
    case IResource::Type::Texture2D:
        {
            return Slang::Math::Max(size.width, size.height);
        }
    default:
        return 0;
    }
}

/// Given the type, calculates the number of mip maps. 0 on error
inline int calcNumMipLevels(IResource::Type type, ITextureResource::Extents size)
{
    const int maxDimensionSize = calcMaxDimension(size, type);
    return (maxDimensionSize > 0) ? (Slang::Math::Log2Floor(maxDimensionSize) + 1) : 0;
}
/// Calculate the total number of sub resources. 0 on error.
inline int calcNumSubResources(const ITextureResource::Desc& desc)
{
    const int numMipMaps =
        (desc.numMipLevels > 0) ? desc.numMipLevels : calcNumMipLevels(desc.type, desc.size);
    const int arrSize = (desc.arraySize > 0) ? desc.arraySize : 1;

    switch (desc.type)
    {
    case IResource::Type::Texture1D:
    case IResource::Type::Texture2D:
    case IResource::Type::Texture3D:
        {
            return numMipMaps * arrSize;
        }
    case IResource::Type::TextureCube:
        {
            // There are 6 faces to a cubemap
            return numMipMaps * arrSize * 6;
        }
    default:
        return 0;
    }
}

IBufferResource::Desc fixupBufferDesc(const IBufferResource::Desc& desc);
ITextureResource::Desc fixupTextureDesc(const ITextureResource::Desc& desc);

Format srgbToLinearFormat(Format format);

} // namespace gfx
