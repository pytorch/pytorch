// unit-test-string-escape.cpp

#include "../../source/core/slang-string-escape-util.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

static bool _checkConversion(StringEscapeHandler* handler, const UnownedStringSlice& check)
{
    StringBuilder buf;
    handler->appendEscaped(check, buf);

    StringBuilder decode;
    handler->appendUnescaped(buf.getUnownedSlice(), decode);

    return decode == check;
}

static bool _checkDecode(const UnownedStringSlice& encoded, const UnownedStringSlice& decoded)
{
    auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);

    StringBuilder buf;
    StringEscapeUtil::appendUnquoted(handler, encoded, buf);
    return buf == decoded;
}

#define SLANG_ENCODED_DECODED(x)      \
    const auto encoded = toSlice(#x); \
    const auto decoded = toSlice(x);

SLANG_UNIT_TEST(StringEscape)
{
    // Check greedy hex digits
    {
        // \x can have any number of hex digits
        const char text[] = "\x000001";
        SLANG_ASSERT(SLANG_COUNT_OF(text) == 2 && text[0] == 1);
    }

    // Check octal greedy
    {
        //\ + up to 3 octal digits
        const char text[] = "\0011";
        SLANG_ASSERT(SLANG_COUNT_OF(text) == 3 && text[0] == 1 && text[1] == '1');

        const char text2[] = "\78";
        SLANG_ASSERT(SLANG_COUNT_OF(text2) == 3 && text2[0] == 7 && text2[1] == '8');
    }

    {
        auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);

        SLANG_CHECK(_checkConversion(
            handler,
            toSlice("\0\1\2"
                    "2")));
    }

    {
        auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);

        // We can't just use '\uxxxx', because it has to be translatable into an output character in
        // MSVC (not into utf8) Can make work perhaps with something like #pragma
        // execution_character_set("utf-8") But for now we don't worry
        //
        // Visual Studio does not appear to support '\U' by default, presumably because wchar_t is
        // 16 bits

        {
            SLANG_ENCODED_DECODED("\a\b\0hey~\u0023\n\0");
            SLANG_CHECK(_checkDecode(encoded, decoded));
        }

        {
            SLANG_ENCODED_DECODED("\n\v\b\t\1\02\003\x5z\x00007f\0");
            SLANG_CHECK(_checkDecode(encoded, decoded));
        }
    }
}
