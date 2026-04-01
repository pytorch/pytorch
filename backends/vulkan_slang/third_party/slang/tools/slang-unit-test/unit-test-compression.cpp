// unit-compression.cpp
#include "../../source/core/slang-deflate-compression-system.h"
#include "../../source/core/slang-lz4-compression-system.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

static ICompressionSystem* _getCompressionSystem(CompressionSystemType type)
{
    switch (type)
    {
    case CompressionSystemType::Deflate:
        return DeflateCompressionSystem::getSingleton();
        break;
    case CompressionSystemType::LZ4:
        return LZ4CompressionSystem::getSingleton();
        break;
    default:
        break;
    }
    return nullptr;
}

SLANG_UNIT_TEST(compression)
{
    // Test out compression systems
    for (Index i = 0; i < Count(CompressionSystemType::CountOf); ++i)
    {
        ICompressionSystem* system = _getCompressionSystem(CompressionSystemType(i));

        if (!system)
        {
            continue;
        }

        const char src[] = "Some text to compress";
        size_t srcSize = sizeof(src);

        ComPtr<ISlangBlob> compressedBlob;

        // Use the default style
        CompressionStyle style;

        SLANG_CHECK(
            SLANG_SUCCEEDED(system->compress(&style, src, srcSize, compressedBlob.writeRef())));

        // Now lets decompress
        List<char> decompressedData;
        decompressedData.setCount(srcSize);

        SLANG_CHECK(SLANG_SUCCEEDED(system->decompress(
            compressedBlob->getBufferPointer(),
            compressedBlob->getBufferSize(),
            srcSize,
            decompressedData.getBuffer())));
        SLANG_CHECK(::memcmp(src, decompressedData.getBuffer(), srcSize) == 0);
    }
}
