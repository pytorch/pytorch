// unit-test-byte-encode.cpp

#include "../../source/core/slang-byte-encode-util.h"
#include "../../source/core/slang-list.h"
#include "../../source/core/slang-random-generator.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

static void checkUInt32(uint32_t value)
{
    uint8_t buffer[ByteEncodeUtil::kMaxLiteEncodeUInt32 + 1];

    int writeLen = ByteEncodeUtil::encodeLiteUInt32(value, buffer);
    buffer[writeLen] = 0xcd;

    uint32_t decode;
    int readLen = ByteEncodeUtil::decodeLiteUInt32(buffer, &decode);

    SLANG_CHECK(readLen == writeLen && decode == value);
}

SLANG_UNIT_TEST(byteEncode)
{
    DefaultRandomGenerator randGen(0x5346536a);

    {
        SLANG_CHECK(ByteEncodeUtil::calcMsb8(0) == -1);
        SLANG_CHECK(ByteEncodeUtil::calcMsb8(1) == 0);
        SLANG_CHECK(ByteEncodeUtil::calcMsb8(0x81) == 7);
    }

    {
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0) == -1);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x81) == 7);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x00000001) == 0);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x00000081) == 7);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x00000181) == 8);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x00008181) == 15);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x00018181) == 16);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x00818181) == 23);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x01818181) == 24);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0x81818181) == 31);
        SLANG_CHECK(ByteEncodeUtil::calcMsb32(0xffffffff) == 31);
    }

    {
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x00000000) == -1);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x00000001) == 0);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x00000081) == 0);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x00000181) == 1);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x00008181) == 1);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x00018181) == 2);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x00818181) == 2);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x01818181) == 3);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0x81818181) == 3);
        SLANG_CHECK(ByteEncodeUtil::calcMsByte32(0xffffffff) == 3);
    }

    {
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x00000001) == 0);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x00000081) == 0);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x00000181) == 1);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x00008181) == 1);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x00018181) == 2);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x00818181) == 2);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x01818181) == 3);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0x81818181) == 3);
        SLANG_CHECK(ByteEncodeUtil::calcNonZeroMsByte32(0xffffffff) == 3);
    }

    {
        const int blockSize = 1024;

        List<uint8_t> encodedBuffer;
        encodedBuffer.setCount(ByteEncodeUtil::kMaxLiteEncodeUInt32 * blockSize);

        List<uint32_t> initialBuffer;
        initialBuffer.setCount(blockSize);
        List<uint32_t> decodeBuffer;
        decodeBuffer.setCount(blockSize);
        // Put in cache?
        memset(decodeBuffer.begin(), 0, blockSize * sizeof(uint32_t));

        for (int i = 0; i < blockSize; i++)
        {
            const int v = ByteEncodeUtil::calcMsb8(uint32_t((randGen.nextInt32() & 0xf) | 1));

            // Make the commonality of different numbers that bytes are most common, then shorts
            // etc..
            uint32_t mask;
            switch (v)
            {
            case 0:
                mask = 0xffffffff;
                break;
            case 1:
                mask = 0x00ffffff;
                break;
            case 2:
                mask = 0x0000ffff;
                break;
            case 3:
                mask = 0x000000ff;
                break;
            }

            initialBuffer[i] = randGen.nextInt32() & mask;
        }

        size_t numEncodeBytes = ByteEncodeUtil::encodeLiteUInt32(
            initialBuffer.begin(),
            blockSize,
            encodedBuffer.begin());

        SLANG_CHECK(
            ByteEncodeUtil::calcEncodeLiteSizeUInt32(initialBuffer.begin(), blockSize) ==
            numEncodeBytes);

        size_t numEncodeBytes2 = ByteEncodeUtil::decodeLiteUInt32(
            encodedBuffer.begin(),
            blockSize,
            decodeBuffer.begin());

        SLANG_CHECK(numEncodeBytes2 == numEncodeBytes);

        SLANG_CHECK(
            memcmp(decodeBuffer.begin(), initialBuffer.begin(), sizeof(uint32_t) * blockSize) == 0);
    }

    {
        checkUInt32(uint32_t(0));
        checkUInt32(uint32_t(0x7fffff));
        checkUInt32(uint32_t(0x7fff));
        checkUInt32(uint32_t(0x7f));
        checkUInt32(uint32_t(0x7fffffff));
        checkUInt32(uint32_t(0xffffffff));

#if 1
        for (int64_t i = 0; i < SLANG_INT64(0x100000000); i += 371)
        {
            checkUInt32(uint32_t(i));
        }
#else
        for (int64_t i = 0; i < SLANG_INT64(0x100000000); i += 1)
        {
            checkUInt32(uint32_t(i));
        }
#endif
    }
}
