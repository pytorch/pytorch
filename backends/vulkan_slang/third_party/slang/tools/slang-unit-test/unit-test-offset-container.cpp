// unit-test-offset-container.cpp

#include "../../source/core/slang-offset-container.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

static void _checkEncodeDecode(uint32_t size)
{
    uint8_t encode[OffsetString::kMaxSizeEncodeSize];

    size_t encodeSize = OffsetString::calcEncodedSize(size, encode);

    size_t decodedSize;
    const char* chars = OffsetString::decodeSize((const char*)encode, decodedSize);

    SLANG_CHECK(decodedSize == size);
    SLANG_CHECK(chars - (const char*)encode == encodeSize);
}

namespace
{ // anonymous

struct Root
{
    Offset32Array<Offset32Ptr<OffsetString>> dirs;
    Offset32Ptr<OffsetString> name;
    float value;
};

} // namespace

SLANG_UNIT_TEST(offsetContainer)
{
    _checkEncodeDecode(253);

    for (int64_t i = 0; i < 0x100000000; i += (i / 2) + 1)
    {
        _checkEncodeDecode(uint32_t(i));
    }

    {
        OffsetContainer container;

        const char* strings[] = {
            "Hello",
            "World",
            nullptr,
        };

        {
            auto& base = container.asBase();

            Offset32Ptr<Root> root = container.newObject<Root>();

            auto array = container.newArray<Offset32Ptr<OffsetString>>(SLANG_COUNT_OF(strings));
            for (Int i = 0; i < SLANG_COUNT_OF(strings); ++i)
            {
                base[array[i]] = container.newString(strings[i]);
            }
            base[root]->dirs = array;
        }

        {
            List<uint8_t> copy;
            copy.addRange(container.getData(), container.getDataCount());

            MemoryOffsetBase base;
            base.set(copy.getBuffer(), copy.getCount());

            Root* root = (Root*)(copy.getBuffer() + kStartOffset);

            SLANG_CHECK(root->dirs.getCount() == SLANG_COUNT_OF(strings));

            Int count = root->dirs.getCount();
            for (Int i = 0; i < count; ++i)
            {
                OffsetString* str = base.asRaw(base.asRaw(root->dirs[i]));

                const char* check = strings[i];

                if (check)
                {
                    SLANG_CHECK(str != nullptr);
                    const char* strCstr = str->getCstr();
                    SLANG_CHECK(strcmp(strCstr, check) == 0);
                }
                else
                {
                    SLANG_CHECK(str == nullptr);
                }
            }

            {
                Index index = 0;
                for (const auto v : root->dirs)
                {
                    OffsetString* str = base.asRaw(base.asRaw(v));
                    const char* check = strings[index];
                    if (check)
                    {
                        SLANG_CHECK(str != nullptr);
                        const char* strCstr = str->getCstr();
                        SLANG_CHECK(strcmp(strCstr, check) == 0);
                    }
                    else
                    {
                        SLANG_CHECK(str == nullptr);
                    }

                    index++;
                }
            }
        }
    }
}
