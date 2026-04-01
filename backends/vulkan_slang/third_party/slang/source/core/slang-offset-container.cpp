// slang-offset-container.cpp
#include "slang-offset-container.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OffsetString !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

size_t OffsetString::calcEncodedSize(size_t size, uint8_t encode[kMaxSizeEncodeSize])
{
    SLANG_ASSERT(size <= 0xffffffff);
    if (size <= kSizeBase)
    {
        encode[0] = uint8_t(size);
        return 1;
    }
    // Encode
    int num = 0;
    while (size)
    {
        encode[num + 1] = uint8_t(size);
        size >>= 8;
        num++;
    }

    // It might be one byte past the front, if its < 0x100 but greater than kSizeBase
    SLANG_ASSERT(num >= 1);

    encode[0] = uint8_t(kSizeBase + num);
    return num + 1;
}

/* static */ const char* OffsetString::decodeSize(const char* in, size_t& outSize)
{
    const uint8_t* cur = (const uint8_t*)in;
    if (*cur <= kSizeBase)
    {
        outSize = *cur;
        return in + 1;
    }

    int numBytes = *cur - kSizeBase;
    switch (numBytes)
    {
    case 1:
        {
            outSize = cur[1];
            return in + 2;
        }
    case 2:
        {
            outSize = cur[1] | (uint32_t(cur[2]) << 8);
            return in + 3;
        }
    case 3:
        {
            outSize = cur[1] | (uint32_t(cur[2]) << 8) | (uint32_t(cur[3]) << 16);
            return in + 4;
        }
    case 4:
        {
            outSize = cur[1] | (uint32_t(cur[2]) << 8) | (uint32_t(cur[3]) << 16) |
                      (uint32_t(cur[4]) << 24);
            return in + 5;
        }
    default:
        {
            outSize = 0;
            return nullptr;
        }
    }
}

/* static */ size_t OffsetString::calcAllocationSize(size_t stringSize)
{
    uint8_t encode[kMaxSizeEncodeSize];
    size_t encodeSize = calcEncodedSize(stringSize, encode);
    // Add 1 for terminating 0
    return encodeSize + stringSize + 1;
}

/* static */ size_t OffsetString::calcAllocationSize(const UnownedStringSlice& slice)
{
    return calcAllocationSize(slice.getLength());
}

UnownedStringSlice OffsetString::getSlice() const
{
    size_t size;
    const char* chars = decodeSize(m_sizeThenContents, size);

    return UnownedStringSlice(chars, size);
}

const char* OffsetString::getCstr() const
{
    return getSlice().begin();
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OffsetContainer !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

OffsetContainer::OffsetContainer()
{
    m_capacity = 0;
    m_data = nullptr;

    // We need to allocate some of the first bytes 0 can be used for nullptr.
    allocateAndZero(kStartOffset, 1);
}

OffsetContainer::~OffsetContainer()
{
    if (m_data)
    {
        ::free(m_data);
    }
}

void* OffsetContainer::allocate(size_t size)
{
    return allocate(size, 1);
}

void OffsetContainer::fixAlignment(size_t alignment)
{
    allocate(0, alignment);
}

void* OffsetContainer::allocate(size_t size, size_t alignment)
{
    size_t offset = (m_dataSize + alignment - 1) & ~(alignment - 1);

    if (offset + size > m_capacity)
    {
        const size_t minSize = offset + size;

        size_t calcSize = m_capacity;
        if (calcSize < 2048)
        {
            calcSize = 2048;
        }
        else
        {
            // Expand geometrically, but lets not double in size...
            calcSize = calcSize + (calcSize / 2);
        }

        // We must be at least minSize
        size_t newSize = (calcSize < minSize) ? minSize : calcSize;

        // Reallocate space
        m_data = (uint8_t*)::realloc(m_data, newSize);
        m_capacity = newSize;
    }

    SLANG_ASSERT(offset + size <= m_capacity);

    m_dataSize = offset + size;
    return m_data + offset;
}

void* OffsetContainer::allocateAndZero(size_t size, size_t alignment)
{
    void* data = allocate(size, alignment);
    memset(data, 0, size);
    return data;
}

Offset32Ptr<OffsetString> OffsetContainer::newString(const UnownedStringSlice& slice)
{
    size_t stringSize = slice.getLength();

    uint8_t head[OffsetString::kMaxSizeEncodeSize];
    size_t headSize = OffsetString::calcEncodedSize(stringSize, head);

    size_t allocSize = headSize + stringSize + 1;
    uint8_t* bytes = (uint8_t*)allocate(allocSize);

    ::memcpy(bytes, head, headSize);
    ::memcpy(bytes + headSize, slice.begin(), stringSize);

    // 0 terminate
    bytes[headSize + stringSize] = 0;

    return Offset32Ptr<OffsetString>(getOffset(bytes));
}

Offset32Ptr<OffsetString> OffsetContainer::newString(const char* contents)
{
    Offset32Ptr<OffsetString> relString;
    if (contents)
    {
        relString = newString(UnownedStringSlice(contents));
    }
    return relString;
}

} // namespace Slang
