// unit-test-riff.cpp

#include "../../source/core/slang-random-generator.h"
#include "../../source/core/slang-riff.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

static void _writeRandom(
    RandomGenerator* rand,
    size_t maxSize,
    RiffContainer& ioContainer,
    List<uint8_t>& ioData)
{
    while (true)
    {
        const Index oldCount = ioData.getCount();

        const size_t allocSize = size_t(rand->nextInt32InRange(1, 50));

        if (allocSize + oldCount > maxSize)
        {
            break;
        }

        ioData.setCount(oldCount + Index(allocSize));
        rand->nextData(ioData.getBuffer() + oldCount, allocSize);

        // Write
        ioContainer.write(ioData.getBuffer() + oldCount, allocSize);
    }

    // Should be a single block with same data as the List
    RiffContainer::DataChunk* dataChunk =
        as<RiffContainer::DataChunk>(ioContainer.getCurrentChunk());
    SLANG_ASSERT(dataChunk);
}

SLANG_UNIT_TEST(riff)
{
    typedef RiffContainer::ScopeChunk ScopeChunk;
    typedef RiffContainer::Chunk::Kind Kind;

    const FourCC markThings = SLANG_FOUR_CC('T', 'H', 'I', 'N');
    const FourCC markData = SLANG_FOUR_CC('D', 'A', 'T', 'A');

    {
        RiffContainer container;

        {
            ScopeChunk scopeContainer(&container, Kind::List, markThings);
            {
                ScopeChunk scopeChunk(&container, Kind::Data, markData);

                const char hello[] = "Hello ";
                const char world[] = "World!";

                container.write(hello, sizeof(hello));
                container.write(world, sizeof(world));
            }

            {
                ScopeChunk scopeChunk(&container, Kind::Data, markData);

                const char test0[] = "Testing... ";
                const char test1[] = "Testing!";

                container.write(test0, sizeof(test0));
                container.write(test1, sizeof(test1));
            }

            {
                ScopeChunk innerScopeContainer(&container, Kind::List, markThings);

                {
                    ScopeChunk scopeChunk(&container, Kind::Data, markData);

                    const char another[] = "Another?";
                    container.write(another, sizeof(another));
                }
            }
        }

        SLANG_CHECK(container.isFullyConstructed());
        SLANG_CHECK(RiffContainer::isChunkOk(container.getRoot()));

        {
            StringBuilder builder;
            {
                StringWriter writer(&builder, 0);
                RiffUtil::dump(container.getRoot(), &writer);
            }

            {
                OwnedMemoryStream stream(FileAccess::ReadWrite);
                SLANG_CHECK(SLANG_SUCCEEDED(RiffUtil::write(container.getRoot(), true, &stream)));

                stream.seek(SeekOrigin::Start, 0);

                RiffContainer readContainer;
                SLANG_CHECK(SLANG_SUCCEEDED(RiffUtil::read(&stream, readContainer)));

                // Dump the read contents
                StringBuilder readBuilder;
                {
                    StringWriter writer(&readBuilder, 0);
                    RiffUtil::dump(readContainer.getRoot(), &writer);
                }

                // They should be the same
                SLANG_CHECK(readBuilder == builder);
            }
        }
    }

    // Test writing as a stream only allocates a single data block (as long as there is enough
    // space).
    {
        RiffContainer container;

        ScopeChunk scopeChunk(&container, Kind::List, markData);
        {
            ScopeChunk scopeChunk(&container, Kind::Data, markData);
            RefPtr<RandomGenerator> rand = RandomGenerator::create(0x345234);

            List<uint8_t> data;
            _writeRandom(
                rand,
                container.getMemoryArena().getBlockPayloadSize() / 2,
                container,
                data);

            // Should be a single block with same data as the List
            RiffContainer::DataChunk* dataChunk =
                as<RiffContainer::DataChunk>(container.getCurrentChunk());
            SLANG_ASSERT(dataChunk);

            // It should be a single block
            SLANG_CHECK(dataChunk->getSingleData() != nullptr);

            SLANG_CHECK(dataChunk->isEqual(data.getBuffer(), data.getCount()));
        }
    }

    // Test writing across multiple data blocks
    {
        RefPtr<RandomGenerator> rand = RandomGenerator::create(0x345234);

        for (Int i = 0; i < 100; ++i)
        {
            RiffContainer container;

            const size_t maxSize = rand->nextInt32InRange(
                1,
                int32_t(container.getMemoryArena().getBlockPayloadSize() * 3));

            ScopeChunk scopeChunk(&container, Kind::List, markData);
            {
                ScopeChunk scopeChunk(&container, Kind::Data, markData);

                List<uint8_t> data;
                _writeRandom(rand, maxSize, container, data);

                // Should be a single block with same data as the List
                RiffContainer::DataChunk* dataChunk =
                    as<RiffContainer::DataChunk>(container.getCurrentChunk());
                SLANG_CHECK(dataChunk && dataChunk->isEqual(data.getBuffer(), data.getCount()));
            }
        }
    }

#if 0
    {
        RiffContainer container;
        {
            FileStream readStream("ambient-drop.wav", FileMode::Open, FileAccess::Read, FileShare::ReadWrite);
            SLANG_CHECK(SLANG_SUCCEEDED(RiffUtil::read(&readStream, container)));
            RiffUtil::dump(container.getRoot(), StdWriters::getOut());
        }
        // Write it
        {

            FileStream writeStream("check.wav", FileMode::Create, FileAccess::Write, FileShare::ReadWrite);
            SLANG_CHECK(SLANG_SUCCEEDED(RiffUtil::write(container.getRoot(), true, &writeStream)));
        }
    }
#endif
}
