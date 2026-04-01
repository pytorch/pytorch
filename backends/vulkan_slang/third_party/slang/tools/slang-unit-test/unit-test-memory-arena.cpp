// unit-test-free-list.cpp

#include "../../source/core/slang-list.h"
#include "../../source/core/slang-memory-arena.h"
#include "../../source/core/slang-random-generator.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;


namespace // anonymous
{

struct Block
{
    void* m_data;
    size_t m_size;
    uint8_t m_value;
};

enum class TestMode
{
    eUnaligned,
    eImplicitAligned, ///< Alignment is kept implicitly with Unaligned allocs of the right size
    eDefaultAligned,
    eExplicitAligned,
    eCount,
};

} // namespace

static size_t getAlignment(TestMode mode)
{
    switch (mode)
    {
    default:
    case TestMode::eUnaligned:
        return 1;
    case TestMode::eExplicitAligned:
        return 16;
    case TestMode::eImplicitAligned:
        return 32;
    case TestMode::eDefaultAligned:
        return MemoryArena::kMinAlignment;
    }
}

static bool hasValueShort(const uint8_t* data, size_t size, uint8_t value)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (data[i] != value)
        {
            return false;
        }
    }
    return true;
}

static bool hasValue(const uint8_t* data, size_t size, uint8_t value)
{
    const size_t alignMask = sizeof(size_t) - 1;

    if (size <= sizeof(size_t) * 2)
    {
        return hasValueShort(data, size, value);
    }

    if (size_t(data) & alignMask)
    {
        size_t firstSize = sizeof(size_t) - (size_t(data) & alignMask);
        if (!hasValueShort(data, firstSize, value))
        {
            return false;
        }
        size -= firstSize;
        data += firstSize;

        assert((size_t(data) & alignMask) == 0);
    }

    // Now do the middle
    size_t numWords = size / sizeof(size_t);

    // Expand the byte up to a word size
    size_t wordValue = (size_t(value) << 8) | value;
    wordValue = (wordValue << 16) | wordValue;
    wordValue = (sizeof(size_t) > 4) ? size_t((uint64_t(wordValue) << 32) | wordValue) : wordValue;

    const size_t* wordData = (const size_t*)data;
    for (size_t i = 0; i < numWords; ++i)
    {
        if (wordData[i] != wordValue)
        {
            return false;
        }
    }

    // Do the end piece
    return hasValueShort(data + sizeof(size_t) * numWords, size & alignMask, value);
}

SLANG_UNIT_TEST(memoryArena)
{
    DefaultRandomGenerator randGen(0x5346536a);

    {
        const size_t blockSize = 1024;
        MemoryArena arena;
        arena.init(blockSize);

        List<void*> blocks;

        blocks.add(arena.allocate(100));
        blocks.add(arena.allocate(blockSize * 2));
        blocks.add(arena.allocate(100));
        blocks.add(arena.allocate(blockSize * 2));
        blocks.add(arena.allocate(100));

        arena.deallocateAll();
        blocks.add(arena.allocate(100));
        blocks.add(arena.allocate(blockSize * 2));

        arena.reset();

        {
            uint32_t data[] = {1, 2, 3};

            const uint32_t* copy = arena.allocateAndCopyArray(data, SLANG_COUNT_OF(data));

            SLANG_CHECK(::memcmp(copy, data, sizeof(data)) == 0);
        }
    }

    {
        int count = 0;
        const size_t blockSize = 1024;

        for (TestMode mode = TestMode(0); int(mode) < int(TestMode::eCount);
             mode = TestMode(int(mode) + 1))
        {
            const size_t alignment = getAlignment(mode);

            MemoryArena arena;
            arena.init(blockSize, alignment);

            List<Block> blocks;

            for (int i = 0; i < 10000; i++)
            {
                count++;

                const int var = randGen.nextInt32() & 0x3ff;
                if (var < 3 && blocks.getCount() > 0)
                {
                    if (var == 1)
                    {
                        // Deallocate everything
                        arena.deallocateAll();
                        blocks.clear();
                    }
                    else if (var == 2)
                    {
                        arena.reset();
                        blocks.clear();
                    }
                    else if (var == 3)
                    {
                        arena.rewindToCursor(nullptr);
                        blocks.clear();
                    }
                    else if (var == 4)
                    {
                        // Rewind to a random position
                        int rewindIndex = randGen.nextInt32UpTo(int32_t(blocks.getCount()));
                        // rewind to this block
                        arena.rewindToCursor(blocks[rewindIndex].m_data);
                        // All the blocks (includign this one) and now deallocated
                        blocks.setCount(rewindIndex);
                    }
                    else
                    {
                        size_t usedMemory = arena.calcTotalMemoryUsed();
                        size_t allocatedMemory = arena.calcTotalMemoryAllocated();

                        SLANG_CHECK(allocatedMemory >= usedMemory);
                    }
                }
                else
                {
                    size_t sizeInBytes = (randGen.nextInt32() & 255) + 1;

                    // Lets go for an oversized block
                    if ((randGen.nextInt32() & 0xff) < 2)
                    {
                        sizeInBytes += blockSize;
                    }
                    else if ((randGen.nextInt32() & 0xff) < 2)
                    {
                        // Let's try for a block that's awkwardly sized
                        sizeInBytes = blockSize / 3 + 10;
                    }

                    const uint8_t value = uint8_t(randGen.nextInt32());

                    void* mem = nullptr;
                    switch (mode)
                    {
                    default:
                    case TestMode::eUnaligned:
                        {
                            mem = arena.allocateUnaligned(sizeInBytes);
                            break;
                        }
                    case TestMode::eImplicitAligned:
                        {
                            // Fix the size to get implicit alignment
                            sizeInBytes = (sizeInBytes & ~(alignment - 1)) + alignment;
                            mem = arena.allocateUnaligned(sizeInBytes);
                            break;
                        }
                    case TestMode::eExplicitAligned:
                        {
                            mem = arena.allocateAligned(sizeInBytes, alignment);
                            break;
                        }
                    case TestMode::eDefaultAligned:
                        {
                            mem = arena.allocate(sizeInBytes);
                            break;
                        }
                    }

                    // Check it is aligned
                    SLANG_CHECK((size_t(mem) & (alignment - 1)) == 0);

                    ::memset(mem, value, sizeInBytes);

                    Block block;

                    block.m_data = mem;
                    block.m_size = sizeInBytes;
                    block.m_value = value;

                    blocks.add(block);
                }

                // Check the blocks
                for (Index j = 0; j < blocks.getCount(); ++j)
                {
                    const Block& block = blocks[j];

                    SLANG_CHECK(arena.isValid(block.m_data, block.m_size));

                    SLANG_CHECK(hasValue((uint8_t*)block.m_data, block.m_size, block.m_value));
                }
            }
        }
    }
    {
        // Do lots of allocations and test out rewind
    }
}
